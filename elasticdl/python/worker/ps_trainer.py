# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from elasticdl.python.common.constants import Initializer, MetricsDictKey
from elasticdl.python.common.dtypes import dtype_numpy_to_tensor
from elasticdl.python.common.model_utils import (
    find_layer,
    get_non_embedding_trainable_vars,
)
from elasticdl.python.common.tensor_utils import EmbeddingTableInfo, Tensor
from elasticdl.python.elasticdl.feature_column import feature_column
from elasticdl.python.elasticdl.layers.embedding import Embedding
from elasticdl.python.worker.trainer import Trainer

# The default maximum number of a minibatch retry as its results
# (e.g. gradients) are not accepted by master.
DEFAULT_MAX_MINIBATCH_RETRY_NUM = 64

DEFAULT_STEPS_TO_CHECK_RENDEZVOUS = 20


class ParameterServerTrainer(Trainer):
    """Parameter Server Trainer"""

    def __init__(self, model, ps_client, timing, args):
        self._optimizer = model.optimizer
        self._loss = model.loss
        self._model = model
        self._ps_client = ps_client

        if self._ps_client is None:
            raise ValueError(
                "PS channels are not set up under parameter server strategy"
            )
        else:
            self._model_versions_from_ps = [
                -1 for _ in range(self._ps_client.ps_num)
            ]

        self._train_eagerly = False
        self._init_embeddings()

        self._non_embed_grads = {}
        self._evaluation_result = {}

        self._non_embed_grads = {}
        self._evaluation_result = {}
        self._var_created = False
        self._timing = timing
        self._get_model_steps = args.get_model_steps

    def _init_embedding_layer(self):
        """
        Init elasticdl.layers.embedding layer list and assign worker to them
        """
        self._embedding_layers = find_layer(self._model, Embedding)
        for layer in self._embedding_layers:
            layer.set_lookup_embedding_func(
                self._ps_client.pull_embedding_vectors
            )

    def _init_embedding_column(self):
        self._embedding_columns = []
        for layer in self._model.layers:
            if isinstance(layer, tf.keras.layers.DenseFeatures):
                for column in layer._feature_columns:
                    if isinstance(column, feature_column.EmbeddingColumn):
                        self._embedding_columns.append(column)
                        self.logger.info(
                            "Initialize ElasticDL EmbeddingColumn:{}".format(
                                column.name
                            )
                        )

        for column in self._embedding_columns:
            column.set_lookup_embedding_func(
                self._ps_client.pull_embedding_vectors
            )

    def _check_name_conflict_of_embedding_layer_and_column(self):
        if not self._embedding_layers or not self._embedding_columns:
            return

        embedding_layer_name_set = set(
            [layer.name for layer in self._embedding_layers]
        )
        embedding_column_name_set = set(
            [column.name for column in self._embedding_columns]
        )
        conflict_name_set = embedding_column_name_set.union(
            embedding_layer_name_set
        )
        if conflict_name_set:
            raise Exception(
                "Name conflict between embedding layer and column: {}".format(
                    conflict_name_set
                )
            )

    def _init_embeddings(self):
        self._init_embedding_layer()
        self._init_embedding_column()
        self._check_name_conflict_of_embedding_layer_and_column()

        self._report_embedding_info()

        self._need_embedding_layer_check = (
            True
            if self._embedding_layers or self._embedding_columns
            else False
        )

    def _set_tape_for_embedding(self, tape):
        for layer in self._embedding_layers:
            layer.set_tape(tape)
        for column in self._embedding_columns:
            column.set_tape(tape)

    def _reset_embedding(self):
        for layer in self._embedding_layers:
            layer.reset()
        for column in self._embedding_columns:
            column.reset()

    def _update_local_model(self):
        if not self._non_embed_grads:
            return
        # Take care of the order of grads and vars if worker modifies
        # `_non_embed_vars` during training.
        self._optimizer.apply_gradients(
            zip(self._non_embed_grads, self._non_embed_vars.values())
        )
        self._non_embed_grads = None

    def _get_model(self):
        self._timing.start_record_time("get_model")
        # 1. Worker tries to pull dense parameters from the PS, maybe one
        # or more PS instances are uninitialized.
        dense_params, uninit_ps = self._ps_client.pull_dense_parameters(
            [i for i in range(self._ps_client.ps_num)],
            self._model_versions_from_ps,
        )

        # 2. Worker pushes local dense parameters to these PS instances
        # to initialize their partition of parameters.
        if len(uninit_ps) > 0:
            for ps_id in uninit_ps:
                # push variable to ps for initialization
                parameters = [
                    Tensor(name, self._non_embed_vars[name].numpy(), None)
                    for name in self._ps_client.ps_to_parameter[ps_id]
                ]
                self._ps_client.push_dense_parameters(
                    parameters, ps_id, self._model_versions_from_ps[ps_id]
                )

            ps_params, uninit = self._ps_client.pull_dense_parameters(
                uninit_ps, self._model_versions_from_ps
            )
            if len(uninit) > 0:
                # TODO: support PS fault-tolerance
                raise RuntimeError("PS initialization failed")
            dense_params.update(ps_params)

        # 3. Assign parameters to local model
        for k, v in dense_params.items():
            self._non_embed_vars[k].assign(v)

        self._model_version = max(self._model_versions_from_ps)
        self._timing.end_record_time("get_model")

    def _report_embedding_info(self):
        # TODO(qijun): only support float32
        infos = []
        if self._embedding_layers:
            for layer in self._embedding_layers:
                infos.append(
                    EmbeddingTableInfo(
                        layer.embedding_weight_name,
                        layer.output_dim,
                        layer.embeddings_initializer,
                        dtype_numpy_to_tensor(np.dtype("float32")),
                    )
                )

        if self._embedding_columns:
            for column in self._embedding_columns:
                # TODO(brightcoder01): The initializer in embedding column is
                # a variable initializer function. For embedding layer, it's a
                # tf.keras.initializers. Keep aligned between these two.
                infos.append(
                    EmbeddingTableInfo(
                        column.embedding_weight_name,
                        column.dimension,
                        Initializer.UNIFORM,
                        dtype_numpy_to_tensor(np.dtype("float32")),
                    )
                )

        self._ps_client.push_embedding_table_infos(infos)

    def _collect_edl_embedding_name_values(self):
        """
        Collect the information of ElasticDL customized
        embeddings such as EDL embedding layer and EDL embedding column.
        Return:
            An array of key-value pair.
            Key is embedding names, layer name for embedding layer
            and column name for embedding column.
            Value is the EmbeddingAndIds tuple.
        """

        embedding_name_values = []
        for layer in self._embedding_layers:
            embedding_name_values.append(
                (layer.embedding_weight_name, layer.embedding_and_ids)
            )
        for column in self._embedding_columns:
            embedding_name_values.append(
                (column.embedding_weight_name, column.embedding_and_ids)
            )

        return embedding_name_values

    def _report_gradient(self, gradients):
        self._timing.start_record_time("report_gradient")

        grads = []
        for i, v in enumerate(self._non_embed_vars.values()):
            if isinstance(gradients[i], tf.IndexedSlices):
                grad = Tensor(
                    v.name,
                    gradients[i].values.numpy(),
                    gradients[i].indices.numpy(),
                )
            else:
                grad = Tensor(v.name, gradients[i].numpy(), None)
            grads.append(grad)

        edl_grads = []
        edl_embedding_name_values = self._collect_edl_embedding_name_values()
        if edl_embedding_name_values:
            non_embed_vars_n = len(self._non_embed_vars)
            edl_embedding_grads = gradients[non_embed_vars_n:]
            bet_number = 0
            for name, embedding_and_ids in edl_embedding_name_values:

                for i in range(bet_number):
                    grad = Tensor(
                        name,
                        edl_embedding_grads[i + bet_number].values.numpy(),
                        edl_embedding_grads[i + bet_number].indices.numpy(),
                    )
                    edl_grads.append(grad)
                bet_number += len(embedding_and_ids)
            if len(edl_embedding_grads) != bet_number:
                raise ValueError(
                    "elasticdl.layers.embedding related gradient number %d "
                    "does not match the number of its output tensor %d."
                    % (len(edl_embedding_grads), bet_number)
                )
        learning_rate = K.get_value(self._model.optimizer.lr)
        accepted, max_version = self._ps_client.push_gradients(
            grads, edl_grads, learning_rate, self._model_versions_from_ps,
        )
        self._timing.end_record_time("report_gradient")
        return accepted, max_version

    def report_prediction_outputs(self, predictions):
        if self._prediction_outputs_processor:
            self._prediction_outputs_processor.process(
                predictions, self._worker_id
            )
        else:
            self.logger.warning(
                "prediction_outputs_processor is not "
                "defined in the model definition. Prediction outputs "
                "are not processed."
            )
        return True

    def get_evaluation_result(self):
        return self._evaluation_result

    def reset_evaluation_result(self):
        self._evaluation_result = {}

    def get_model_version(self):
        return self._model_version

    def init_variables_if_need(self, features):
        if self._need_embedding_layer_check or not self._var_created:
            self._run_model_call_before_training(features)

    def _run_model_call_before_training(self, features):
        """Call `self._model.call` before training for two things:
            * Create variables and report to ps if not created.
            * Check whether there is an embedding layer that is called
              more than once during one forward-pass.
        """
        if self._embedding_layers:
            with tf.GradientTape() as tape:
                self._set_tape_for_embedding(tape)
                _ = self._model.call(features)
        else:
            _ = self._model.call(features)
        self._non_embed_vars = {}
        for var in get_non_embedding_trainable_vars(
            self._model, self._embedding_layers
        ):
            self._non_embed_vars[var.name] = var

        self._var_created = True

        self._ps_client.partition_dense_parameters(self._non_embed_vars.keys())

        if self._need_embedding_layer_check:
            self._train_eagerly = False
            for layer in self._embedding_layers:
                if len(layer.embedding_and_ids) > 1:
                    self._train_eagerly = True
                    self.logger.warning(
                        "ElasticDL embedding layer %s is called more than "
                        "once, this will make the training process unable "
                        "to accelerate with tf.function." % (layer.name)
                    )
            self._need_embedding_layer_check = False

        self._reset_embedding()

    def get_trainable_items(self):
        """
        return all trainable variables list, including batch embedding
        tensor (BET) if exists. take care to keep the same order as in
        self.report_gradient()
        """
        bets = []
        if self._embedding_layers:
            for layer in self._embedding_layers:
                bets.extend(
                    [
                        batch_embedding
                        for (batch_embedding, _) in layer.embedding_and_ids
                    ]
                )

        if self._embedding_columns:
            for column in self._embedding_columns:
                bets.extend(
                    [
                        batch_embedding
                        for (batch_embedding, _) in column.embedding_and_ids
                    ]
                )

        return list(self._non_embed_vars.values()) + bets

    def train_minibatch(self, features, labels, train_with_local_model=False):
        """
        training for models with elasticdl.layers.embedding does not
        support tf.function decorator
        """
        if not train_with_local_model:
            self._get_model()
        if self._train_eagerly:
            loss, grads = self._training_process_eagerly(features, labels)
        else:
            loss, grads = self._training_process_with_acceleration(
                features, labels
            )

        return (*self._update_global_model(grads), loss)

    @tf.function
    def _training_process_with_acceleration(self, features, labels):
        return self._training_process_eagerly(features, labels)

    def _training_process_eagerly(self, features, labels):
        with tf.GradientTape() as tape:
            self._set_tape_for_embedding(tape)
            outputs = self._model.call(features, training=True)
            loss = self._loss(labels, outputs)
            # Add regularization loss if any
            if self._model.losses:
                loss += tf.math.add_n(self._model.losses)
        grads = tape.gradient(loss, self.get_trainable_items())
        return loss, grads

    @tf.function
    def _forward_process(self, features):
        """Calculates model outputs in non-training mode."""
        outputs = self._model.call(features, training=False)
        return outputs

    def _update_global_model(self, grads):
        accepted, min_model_version = self._report_gradient(grads)
        if accepted and self._get_model_steps > 1:
            non_embed_vars_n = len(self._non_embed_vars)
            self._non_embed_grads = grads[:non_embed_vars_n]
        self._reset_embedding()
        return accepted, min_model_version

    def _collect_evaluation_result(self, outputs, labels):
        key = MetricsDictKey.MODEL_OUTPUT
        if key not in self._evaluation_result:
            outputs = {k: [v.numpy()] for k, v in outputs.items()}
            self._evaluation_result[key] = outputs
        else:
            for k, v in outputs.items():
                self._evaluation_result[key][k].append(v.numpy())
        key = MetricsDictKey.LABEL
        if key not in self._evaluation_result:
            self._evaluation_result[key] = [labels.numpy()]
        else:
            self._evaluation_result[key].append(labels.numpy())

    def evaluate_minibatch(self, features, labels):
        outputs = self._forward_process(features)
        if not isinstance(outputs, dict):
            outputs = {MetricsDictKey.MODEL_OUTPUT: outputs}
        self._collect_evaluation_result(outputs, labels)

    def predict_minibatch(self, features, min_model_version):
        if self._model_version != min_model_version:
            self.get_model()
        predictions = self._forward_process(features)
        return self.report_prediction_outputs(predictions)
