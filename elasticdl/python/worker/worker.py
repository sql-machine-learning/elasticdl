import traceback

import numpy as np
import tensorflow as tf

from elasticdl.proto import elasticdl_pb2, elasticdl_pb2_grpc
from elasticdl.python.common.constants import JobType, MetricsDictKey, Mode
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.common.model_handler import ModelHandler
from elasticdl.python.common.model_utils import (
    find_layer,
    get_dict_from_params_str,
    get_model_spec,
    get_non_embedding_trainable_vars,
)
from elasticdl.python.common.ndarray import (
    ndarray_to_tensor,
    tensor_to_ndarray,
)
from elasticdl.python.elasticdl.layers.embedding import Embedding
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
        data_reader_params="",
        prediction_outputs_processor="PredictionOutputsProcessor",
        max_minibatch_retry_num=DEFAULT_MAX_MINIBATCH_RETRY_NUM,
        get_model_steps=1,
        distribution_strategy=None,
    ):
        """
        Arguments:
            worker_id: The worker ID.
            job_type: The job type.
            minibatch_size: The size of the minibatch used for each iteration.
            model_zoo: The directory that contains user-defined model files
                or a specific model file.
            dataset_fn: The name of the dataset function defined in the
                model file.
            loss: The name of the loss function defined in the model file.
            optimizer: The name of the optimizer defined in the model file.
            eval_metrics_fn: The name of the evaluation metrics function
                defined in the model file.
            channel: The channel for the gRPC master service.
            embedding_service_endpoint: The endpoint to the embedding service.
            model_def: The import path to the model definition
                function/class in the model zoo, e.g.
                "cifar10_subclass.CustomModel".
            model_params: The dictionary of model parameters in a string
                separated by semi-colon used to instantiate the model,
                e.g. "param1=1; param2=2".
            data_reader_params: The data reader parameters in a string
                separated by semi-colon used to instantiate the data reader,
                e.g. "param1=1; param2=2".
            prediction_outputs_processor: The name of the prediction output
                processor class defined in the model file.
            get_model_steps: Worker will perform `get_model` from the
                parameter server every this many steps.
            max_minibatch_retry_num: The maximum number of a minibatch retry
                as its results (e.g. gradients) are not accepted by master.
        """
        self._worker_id = worker_id
        self._job_type = job_type
        self._minibatch_size = minibatch_size
        (
            model_inst,
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
        model_handler = ModelHandler.get_model_handler(distribution_strategy)
        model_inst = model_handler.get_model_to_train(model_inst)

        self._embedding_service_endpoint = embedding_service_endpoint
        self.set_model(model_inst)

        if channel is None:
            self._stub = None
        else:
            self._stub = elasticdl_pb2_grpc.MasterStub(channel)
        self._max_minibatch_retry_num = max_minibatch_retry_num
        self._model_version = -1
        self._task_data_service = TaskDataService(
            self,
            self._job_type == JobType.TRAINING_WITH_EVALUATION,
            data_reader_params=get_dict_from_params_str(data_reader_params),
        )
        self._get_model_steps = get_model_steps
        if self._get_model_steps > 1:
            self._opt = self._opt_fn()
            self._non_embed_grads = None
        self._evaluation_result = {}

    # TODO: Multiple tests are currently using this function to initialize
    # self._model, where the initialization should be done via constructor.
    def set_model(self, model_inst):
        """Set model instance to worker."""
        self._model = model_inst
        self._train_eagerly = False
        self._init_embedding_layer()
        self._var_created = self._model.built
        self._non_embed_vars = {}
        if self._var_created:
            for var in get_non_embedding_trainable_vars(
                self._model, self._embedding_layers
            ):
                self._non_embed_vars[var.name] = var

    def _init_embedding_layer(self):
        """
        Init elasticdl.layers.embedding layer list and assign worker to them
        """
        self._embedding_layers = find_layer(self._model, Embedding)
        for layer in self._embedding_layers:
            layer.set_endpoint(self._embedding_service_endpoint)
        self._need_embedding_layer_check = (
            True if self._embedding_layers else False
        )

    def _set_tape_for_embedding(self, tape):
        for layer in self._embedding_layers:
            layer.set_tape(tape)

    def _reset_embedding(self):
        for layer in self._embedding_layers:
            layer.reset()

    def _update_local_model(self):
        if not self._non_embed_grads:
            return
        # Take care of the order of grads and vars if worker modifies
        # `_non_embed_vars` during training.
        self._opt.apply_gradients(
            zip(self._non_embed_grads, self._non_embed_vars.values())
        )
        self._non_embed_grads = None

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

        # Assumes all trainable variables exist in model.param.
        for tensor_pb in model.param:
            self._non_embed_vars[tensor_pb.name].assign(
                tensor_to_ndarray(tensor_pb)
            )
        self._model_version = model.version

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
        for v in self._non_embed_vars.values():
            req.variable.append(ndarray_to_tensor(v.numpy(), v.name))
        self._stub.ReportVariable(req)

    def report_gradient(self, grads):
        """
        report gradient to ps, return (accepted, model_version) from rpc call.
        """
        req = elasticdl_pb2.ReportGradientRequest()
        non_embed_vars_n = len(self._non_embed_vars)
        # The first `non_embed_vars_n` items in `grads` are gradients for
        # `self._non_embed_vars`.
        # Take care of the order of grads and vars if worker modifies
        # `_non_embed_vars` during training.
        for g, v in zip(
            grads[:non_embed_vars_n], self._non_embed_vars.values()
        ):
            if isinstance(g, tf.IndexedSlices):
                req.gradient.append(
                    ndarray_to_tensor(
                        g.values.numpy(), v.name, tuple(g.indices.numpy())
                    )
                )
            else:
                req.gradient.append(ndarray_to_tensor(g.numpy(), v.name))

        # Accumulate gradients of ElasticDL embedding layer
        if self._embedding_layers:
            # The `edl_embedding_grads` are gradients for bets in
            # `self._embedding_layers`
            edl_embedding_grads = grads[non_embed_vars_n:]

            # Check that the number of bet equal to the number of gradients.
            # Please note that every embedding layer may have more than one
            # `bet_id_pair`.
            bet_number = 0
            for layer in self._embedding_layers:
                bet_number += len(layer.embedding_and_ids)
            if len(edl_embedding_grads) != bet_number:
                raise ValueError(
                    "elasticdl.layers.embedding related gradient number %d "
                    "does not match the number of its output tensor %d."
                    % (len(edl_embedding_grads), bet_number)
                )

            grad_accum_iter = 0
            for layer in self._embedding_layers:
                g_values = None
                g_indices = None
                for _, ids in layer.embedding_and_ids:
                    grad = edl_embedding_grads[grad_accum_iter]
                    grad_accum_iter += 1
                    # ElasticDL embedding layer with Sparse Gradients
                    if isinstance(grad, tf.IndexedSlices):
                        grad = grad.values
                    if g_values is not None:
                        g_values = tf.concat([g_values, grad], axis=0)
                        g_indices = tf.concat([g_indices, ids], axis=0)
                    else:
                        g_values = grad
                        g_indices = ids

                req.gradient.append(
                    ndarray_to_tensor(
                        g_values.numpy(), layer.name, tuple(g_indices.numpy())
                    )
                )

        req.model_version = self._model_version
        res = self._stub.ReportGradient(req)
        return res.accepted, res.model_version

    def report_evaluation_metrics(self, model_outputs, labels):
        """
        report evaluation metrics to ps, return (accepted, model_version)
        from rpc call.
        """
        req = elasticdl_pb2.ReportEvaluationMetricsRequest()
        for name, output in model_outputs.items():
            output = np.concatenate(output)
            req.model_outputs.append(ndarray_to_tensor(output, name))
        labels = np.concatenate(labels)
        req.labels.CopyFrom(ndarray_to_tensor(labels))
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

        if not self._var_created:
            self.report_variable()
            self._var_created = True

        if self._need_embedding_layer_check:
            self._train_eagerly = False
            for layer in self._embedding_layers:
                if len(layer.embedding_and_ids) > 1:
                    self._train_eagerly = True
                    logger.warning(
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
        return list(self._non_embed_vars.values()) + bets

    def training_process(self, features, labels):
        """
        training for models with elasticdl.layers.embedding does not
        support tf.function decorator
        """
        if self._train_eagerly:
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
    def forward_process(self, features):
        """Calculates model outputs in non-training mode."""
        outputs = self._model.call(features, training=False)
        return outputs

    def _run_training_task(self, features, labels):
        loss, grads = self.training_process(features, labels)
        accepted, min_model_version = self.report_gradient(grads)
        if accepted and self._get_model_steps > 1:
            non_embed_vars_n = len(self._non_embed_vars)
            self._non_embed_grads = grads[:non_embed_vars_n]
        self._reset_embedding()
        return accepted, min_model_version, loss

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

    def _run_evaluation_task(self, features, labels):
        outputs = self.forward_process(features)
        if not isinstance(outputs, dict):
            outputs = {MetricsDictKey.MODEL_OUTPUT: outputs}
        self._collect_evaluation_result(outputs, labels)
        return True

    def _run_prediction_task(self, features):
        predictions = self.forward_process(features)
        return self.report_prediction_outputs(predictions)

    def _process_minibatch(
        self,
        task_type,
        features,
        labels,
        min_model_version,
        train_with_local_model=False,
    ):
        if self._need_embedding_layer_check or not self._var_created:
            self._run_model_call_before_training(features)
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
                if not train_with_local_model:
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

        Return:
            A python bool indicating whether worker processed some evaluation
            tasks.
        """
        eval_info = self._task_data_service.get_validation_dataset()
        if not eval_info:
            return False
        (eval_dataset, model_version, task_id) = eval_info
        eval_dataset = self._dataset_fn(
            eval_dataset,
            Mode.EVALUATION,
            self._task_data_service.data_reader.metadata,
        )
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
        accepted, _ = self.report_evaluation_metrics(
            self._evaluation_result[MetricsDictKey.MODEL_OUTPUT],
            self._evaluation_result[MetricsDictKey.LABEL],
        )
        if not accepted:
            raise RuntimeError("Report evaluation metric failed!")
        self.report_task_result(task_id, err_msg)
        self._evaluation_result = {}
        return True

    def _process_save_model_task_if_needed(self):
        task, dataset = (
            self._task_data_service.get_save_model_task_and_dataset()
        )
        if task and dataset:
            # TODO: Implement the save model execution process
            return

    def _process_minibatch_and_report(
        self,
        dataset_batch,
        task_type,
        model_version,
        train_with_local_model=False,
    ):
        err_msg = ""
        try:
            if self._job_type == JobType.PREDICTION_ONLY:
                features = dataset_batch
                labels = None
            else:
                features = dataset_batch[0]
                labels = dataset_batch[1]
            self._process_minibatch(
                task_type,
                features,
                labels,
                model_version,
                train_with_local_model,
            )
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

        # The worker needs to get model from PS if
        # `train_with_local_model=False`. This happens when:
        #     processing first minibatch
        #     last minibatch is evaluation task
        #     last minibatch is training task and failed
        #     local_update_count >= worker._get_model_steps
        # Otherwise, worker trains with local model, i.e.
        # `train_with_local_model=True`
        train_with_local_model = False

        # Initialize `local_update_count=get_model_steps` in order to set
        # `train_with_local_model` to False inside for-loop for the first
        # minibatch.
        local_update_count = self._get_model_steps
        last_training_minibatch_failed = False
        last_minibatch_do_evaluation = False
        while True:
            dataset = self._task_data_service.get_dataset()
            if not dataset:
                break
            dataset = self._dataset_fn(
                dataset, mode, self._task_data_service.data_reader.metadata
            )
            dataset = dataset.batch(self._minibatch_size).prefetch(1)
            for dataset_batch in dataset:
                if self._job_type == JobType.TRAINING_WITH_EVALUATION:
                    if self._process_eval_task_if_needed():
                        last_minibatch_do_evaluation = True
                task = self._task_data_service.get_current_task()
                if task.type == elasticdl_pb2.TRAINING:
                    if (
                        last_minibatch_do_evaluation
                        or last_training_minibatch_failed
                        or local_update_count >= self._get_model_steps
                    ):
                        local_update_count = 0
                        train_with_local_model = False
                    else:
                        train_with_local_model = True
                err_msg = self._process_minibatch_and_report(
                    dataset_batch,
                    task.type,
                    task.model_version,
                    train_with_local_model,
                )
                if task.type == elasticdl_pb2.TRAINING:
                    last_minibatch_do_evaluation = False
                    local_update_count += 1
                    if err_msg:
                        last_training_minibatch_failed = True
                    else:
                        last_training_minibatch_failed = False
                        if local_update_count < self._get_model_steps:
                            self._update_local_model()
                self._task_data_service.report_record_done(
                    self._minibatch_size, err_msg
                )
            del dataset
            # New evaluation tasks may be created after this worker's
            # training tasks are done, as other workers' may still
            # have pending training tasks.
            if self._job_type == JobType.TRAINING_WITH_EVALUATION:
                if self._process_eval_task_if_needed():
                    last_minibatch_do_evaluation = True

            self._process_save_model_task_if_needed()
