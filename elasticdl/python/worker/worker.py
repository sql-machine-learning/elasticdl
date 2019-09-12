import traceback

import numpy as np
import tensorflow as tf

from elasticdl.proto import elasticdl_pb2, elasticdl_pb2_grpc
from elasticdl.python.common.constants import JobType, Mode
from elasticdl.python.common.log_util import default_logger as logger
from elasticdl.python.common.model_helper import find_layer, get_model_spec
from elasticdl.python.common.ndarray import (
    ndarray_to_tensor,
    tensor_to_ndarray,
)
from elasticdl.python.elasticdl.layers.embedding import Embedding
from elasticdl.python.master.embedding_service import EmbeddingService
from elasticdl.python.worker.task_data_service import TaskDataService

# The default maximum number of a minibatch retry as its results
# (e.g. gradients) are not accepted by master.
DEFAULT_MAX_MINIBATCH_RETRY_NUM = 64


class Worker(object):
    """ElasticDL worker"""

    def __init__(
        self,
        worker_id,
        job_type,
        minibatch_size,
        model_zoo,
        dataset_fn="dataset_fn",
        loss="loss",
        optimizer="optimizer",
        eval_metrics_fn="eval_metrics_fn",
        channel=None,
        embedding_service_endpoint=None,
        model_def=None,
        model_params="",
        prediction_outputs_processor="PredictionOutputsProcessor",
        max_minibatch_retry_num=DEFAULT_MAX_MINIBATCH_RETRY_NUM,
        get_model_steps=1,
    ):
        """
        Arguments:
            model_file: A module to define the model
            channel: grpc channel
            max_minibatch_retry_num: The maximum number of a minibatch retry
                as its results (e.g. gradients) are not accepted by master.
        """
        self._worker_id = worker_id
        self._job_type = job_type
        self._minibatch_size = minibatch_size
        (
            self._model,
            self._dataset_fn,
            self._loss,
            self._opt_fn,
            self._eval_metrics_fn,
            self._prediction_outputs_processor,
        ) = get_model_spec(
            model_zoo=model_zoo,
            model_def=model_def,
            dataset_fn=dataset_fn,
            loss=loss,
            optimizer=optimizer,
            eval_metrics_fn=eval_metrics_fn,
            model_params=model_params,
            prediction_outputs_processor=prediction_outputs_processor,
        )
        self._init_embedding_layer()
        self._var_created = self._model.built
        if self._var_created:
            (
                self._non_embed_vars,
                self._embed_vars
            ) = get_trainable_items(self._model, self._embedding_layers)

        if channel is None:
            self._stub = None
        else:
            self._stub = elasticdl_pb2_grpc.MasterStub(channel)
        self._embedding_service_endpoint = embedding_service_endpoint
        self._max_minibatch_retry_num = max_minibatch_retry_num
        self._model_version = -1
        self._task_data_service = TaskDataService(
            self, self._job_type == JobType.TRAINING_WITH_EVALUATION
        )
        self._get_model_steps = get_model_steps

    def _init_embedding_layer(self):
        """
        Init elasticdl.layers.embedding layer list and assign worker to them
        """
        self._embedding_layers = find_layer(self._model, Embedding)
        for layer in self._embedding_layers:
            layer.set_lookup_func(self.lookup_embedding)
        if self._embedding_layers:
            # TODO check that Redis IP/PORT is set
            pass

    def _set_tape_for_embedding(self, tape):
        for layer in self._embedding_layers:
            layer.set_tape(tape)

    def _reset_embedding(self):
        for layer in self._embedding_layers:
            layer.reset()

    def get_task(self):
        """
        get task from master
        """
        req = elasticdl_pb2.GetTaskRequest()
        req.worker_id = self._worker_id

        return self._stub.GetTask(req)

    def get_model(self, version, method):
        """
        get model from master, and update model_version
        """
        req = elasticdl_pb2.GetModelRequest()
        req.version = version
        req.method = method
        model = self._stub.GetModel(req)

        for var in self._non_embed_vars:
            # Assumes all trainable variables exist in model.param.
            var.assign(tensor_to_ndarray(model.param[var.name]))
        self._model_version = model.version

    def lookup_embedding(
        self, ids, layer_name, initializer="uniform", embedding_table_dim=128
    ):
        keys = [Embedding.get_key([layer_name, id]) for id in ids]
        ES_lookup_embedding = EmbeddingService.lookup_embedding
        embedding_vectors, unknown_keys_index = ES_lookup_embedding(
            keys=keys,
            embedding_service_endpoint=self._embedding_service_endpoint,
        )
        if unknown_keys_index:
            # Initialize unknown_keys' embedding vectors and write into Redis.
            unknown_keys = [keys[index] for index in unknown_keys_index]
            initializer = tf.keras.initializers.get(initializer)
            embedding_vector_init = [
                initializer(shape=[1, embedding_table_dim]).numpy()
                for _ in unknown_keys
            ]
            embedding_vector_init = np.concatenate(
                embedding_vector_init, axis=0
            )
            EmbeddingService.update_embedding(
                keys=unknown_keys,
                embedding_vectors=embedding_vector_init,
                embedding_service_endpoint=self._embedding_service_endpoint,
                set_if_not_exist=True,
            )
            # Lookup unknown_keys' embedding vectors
            embedding_vectors_new, unknown_keys_idx_new = ES_lookup_embedding(
                keys=unknown_keys,
                embedding_service_endpoint=self._embedding_service_endpoint,
            )
            if unknown_keys_idx_new:
                raise Exception(
                    "Update embedding vector: %s failed."
                    % str(
                        [unknown_keys[index] for index in unknown_keys_idx_new]
                    )
                )
            for key_index, vector in zip(
                unknown_keys_index, embedding_vectors_new
            ):
                embedding_vectors[key_index] = vector
        embedding_vectors = np.concatenate(embedding_vectors, axis=0)
        return embedding_vectors.reshape((len(keys), embedding_table_dim))

    def report_task_result(self, task_id, err_msg):
        """
        report task result to master
        """
        report = elasticdl_pb2.ReportTaskResultRequest()
        report.task_id = task_id
        report.err_message = err_msg
        return self._stub.ReportTaskResult(report)

    def report_variable(self):
        """
        report variable to ps.
        """
        req = elasticdl_pb2.ReportVariableRequest()
        for v in self._non_embed_vars:
            req.variable[v.name].CopyFrom(ndarray_to_tensor(v.numpy()))
        self._stub.ReportVariable(req)

    def report_gradient(self, grads):
        """
        report gradient to ps, return (accepted, model_version) from rpc call.
        """
        req = elasticdl_pb2.ReportGradientRequest()
        non_embed_vars_n = len(self._non_embed_vars)
        # should keep the same order as self.get_trainable_items()
        for g, v in zip(grads[:non_embed_vars_n], self._non_embed_vars):
            if isinstance(g, tf.IndexedSlices):
                req.gradient[v.name].CopyFrom(
                    ndarray_to_tensor(
                        g.values.numpy(), tuple(g.indices.numpy())
                    )
                )
            else:
                req.gradient[v.name].CopyFrom(ndarray_to_tensor(g.numpy()))

        # deal with gradients of ElasticDL embedding layer
        # should keep the same order as self.get_trainable_items()
        if self._embedding_layers:
            grads_edlembedding = grads[non_embed_vars_n:]

            bet_number = 0
            for layer in self._embedding_layers:
                bet_number += len(layer.bet_ids_pair)
            if len(grads_edlembedding) != bet_number:
                raise ValueError(
                    "elasticdl.layers.embedding related gradient number %d "
                    "does not match the number of its output tensor %d."
                    % (len(grads_edlembedding), bet_number)
                )

            it = 0
            for layer in self._embedding_layers:
                g_values = None
                g_indices = None
                for bet, ids in layer.bet_ids_pair:
                    grad = grads_edlembedding[it]
                    it += 1
                    # ElasticDL embedding layer with Sparse Gradients
                    if isinstance(grad, tf.IndexedSlices):
                        grad = grad.values
                    if g_values is not None:
                        g_values = tf.concat([g_values, grad], axis=0)
                        g_indices = tf.concat([g_indices, ids], axis=0)
                    else:
                        g_values = grad
                        g_indices = ids

                req.gradient[layer.name].CopyFrom(
                    ndarray_to_tensor(
                        g_values.numpy(), tuple(g_indices.numpy())
                    )
                )

        req.model_version = self._model_version
        res = self._stub.ReportGradient(req)
        return res.accepted, res.model_version

    def report_evaluation_metrics(self, evaluation_metrics):
        """
        report evaluation metrics to ps, return (accepted, model_version)
        from rpc call.
        """
        req = elasticdl_pb2.ReportEvaluationMetricsRequest()
        for k, v in evaluation_metrics.items():
            v_np = v.numpy()
            # If scalar, convert to numpy 1D array with size 1
            if not v_np.shape:
                v_np = v_np.reshape(1)
            req.evaluation_metrics[k].CopyFrom(ndarray_to_tensor(v_np))
        req.model_version = self._model_version
        res = self._stub.ReportEvaluationMetrics(req)
        return res.accepted, res.model_version

    def report_prediction_outputs(self, predictions):
        if self._prediction_outputs_processor:
            self._prediction_outputs_processor.process(
                predictions, self._worker_id
            )
        else:
            logger.warning(
                "prediction_outputs_processor is not "
                "defined in the model definition. Prediction outputs "
                "are not processed."
            )
        return True

    def _create_variable_and_report(self, features):
        # Use model.call to create variables, then report to ps
        _ = self._model.call(features)
        self.report_variable()
        self._var_created = True

    def get_trainable_items(self):
        """
        return all trainable variables list, including batch embedding
        tensor (BET) if exists. take care to keep the same order as in
        self.report_gradient()
        """
        bets = []
        if self._embedding_layers:
            for layer in self._embedding_layers:
                bets.extend([i for (i, _) in layer.bet_ids_pair])
        return self._model.trainable_variables + bets

    def training_process(self, features, labels):
        """
        training for models with elasticdl.layers.embedding does not
        support tf.function decorator
        """
        if self._embedding_layers:
            return self.training_process_eagerly(features, labels)
        else:
            return self.training_process_with_acceleration(features, labels)

    @tf.function
    def training_process_with_acceleration(self, features, labels):
        return self.training_process_eagerly(features, labels)

    def training_process_eagerly(self, features, labels):
        with tf.GradientTape() as tape:
            self._set_tape_for_embedding(tape)
            outputs = self._model.call(features, training=True)
            loss = self._loss(outputs, labels)
            # Add regularization loss if any
            if self._model.losses:
                loss += tf.math.add_n(self._model.losses)
        grads = tape.gradient(loss, self.get_trainable_items())
        return loss, grads

    @tf.function
    def evaluation_process(self, features, labels):
        outputs = self._model.call(features, training=False)
        evaluation_metrics = self._eval_metrics_fn(outputs, labels)
        return evaluation_metrics

    @tf.function
    def predict_process(self, features):
        outputs = self._model.call(features, training=False)
        return outputs

    def _run_training_task(self, features, labels):
        loss, grads = self.training_process(features, labels)
        accepted, min_model_version = self.report_gradient(grads)
        self._reset_embedding()
        return accepted, min_model_version, loss

    def _run_evaluation_task(self, features, labels):
        evaluation_metrics = self.evaluation_process(features, labels)
        accepted, _ = self.report_evaluation_metrics(evaluation_metrics)
        return accepted

    def _run_prediction_task(self, features):
        predictions = self.predict_process(features)
        return self.report_prediction_outputs(predictions)

    def _process_minibatch(
        self, task_type, features, labels, min_model_version
    ):
        if not self._var_created:
            self._create_variable_and_report(features)
        for _ in range(self._max_minibatch_retry_num):
            if task_type == elasticdl_pb2.EVALUATION:
                if min_model_version == -1:
                    if self._model_version < 0:
                        self.get_model(0, elasticdl_pb2.MINIMUM)
                elif self._model_version != min_model_version:
                    self.get_model(min_model_version, elasticdl_pb2.FIXED)
                accepted = self._run_evaluation_task(features, labels)
                if accepted:
                    break
            elif task_type == elasticdl_pb2.TRAINING:
                # TODO: optimize the logic to avoid unnecessary
                #       get_model call.
                self.get_model(
                    max(self._model_version, min_model_version),
                    elasticdl_pb2.MINIMUM,
                )
                accepted, min_model_version, loss = self._run_training_task(
                    features, labels
                )
                if accepted:
                    logger.info("Loss is %f" % loss.numpy())
                    break
            elif task_type == elasticdl_pb2.PREDICTION:
                if self._model_version != min_model_version:
                    self.get_model(min_model_version, elasticdl_pb2.FIXED)
                accepted = self._run_prediction_task(features)
                if accepted:
                    break
            else:
                raise RuntimeError("Unrecognized task type, %s" % task_type)
        else:
            # Worker got stuck, fail the task.
            # TODO: stop the worker if it fails to make any
            #       progress for some time.
            raise RuntimeError("Worker got stuck")
        return min_model_version

    def _process_eval_task_if_needed(self):
        """
        Check if there are evaluation tasks and process the tasks if any.
        """
        eval_info = self._task_data_service.get_evaluation_dataset()
        if not eval_info:
            return
        (eval_dataset, model_version, task_id) = eval_info
        eval_dataset = self._dataset_fn(eval_dataset, Mode.EVALUATION)
        eval_dataset = eval_dataset.batch(self._minibatch_size).prefetch(1)
        err_msg = ""
        for dataset_batch in eval_dataset:
            data_err_msg = self._process_minibatch_and_report(
                dataset_batch, elasticdl_pb2.EVALUATION, model_version
            )
            if data_err_msg:
                err_msg = data_err_msg
                break
        del eval_dataset
        self.report_task_result(task_id, err_msg)

    def _process_minibatch_and_report(
        self, dataset_batch, task_type, model_version
    ):
        err_msg = ""
        try:
            if self._job_type == JobType.PREDICTION_ONLY:
                features = dataset_batch
                labels = None
            else:
                features = dataset_batch[0]
                labels = dataset_batch[1]
            self._process_minibatch(task_type, features, labels, model_version)
        except RuntimeError as err:
            err_msg = str(err)
            traceback.print_exc()
        except Exception as ex:
            err_msg = str(ex)
            traceback.print_exc()
            raise ex
        return err_msg

    def run(self):
        """
        Fetches task from master with and performs training or evaluation.
        """
        if self._job_type == JobType.PREDICTION_ONLY:
            mode = Mode.PREDICTION
        elif self._job_type == JobType.EVALUATION_ONLY:
            mode = Mode.EVALUATION
        else:
            mode = Mode.TRAINING
        while True:
            dataset = self._task_data_service.get_dataset()
            if not dataset:
                break
            dataset = self._dataset_fn(dataset, mode)
            dataset = dataset.batch(self._minibatch_size).prefetch(1)
            for dataset_batch in dataset:
                if self._job_type == JobType.TRAINING_WITH_EVALUATION:
                    self._process_eval_task_if_needed()
                task = self._task_data_service.get_current_task()
                err_msg = self._process_minibatch_and_report(
                    dataset_batch, task.type, task.model_version
                )
                self._task_data_service.report_record_done(
                    self._minibatch_size, err_msg
                )
            del dataset
            # New evaluation tasks may be created after this worker's
            # training tasks are done, as other workers' may still
            # have pending training tasks.
            if self._job_type == JobType.TRAINING_WITH_EVALUATION:
                self._process_eval_task_if_needed()
