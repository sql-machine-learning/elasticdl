import traceback

import tensorflow as tf

from elasticdl.proto import elasticdl_pb2, elasticdl_pb2_grpc
from elasticdl.python.common.constants import JobType, Mode
from elasticdl.python.common.log_util import default_logger as logger
from elasticdl.python.common.model_helper import (
    find_layer,
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
        prediction_outputs_processor="PredictionOutputsProcessor",
        max_minibatch_retry_num=DEFAULT_MAX_MINIBATCH_RETRY_NUM,
        get_model_steps=1,
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
            model_params: The dictionary of model parameters in a string that
                will be used to instantiate the model,
                e.g. "param1=1,param2=2".
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
        self._embedding_service_endpoint = embedding_service_endpoint
        self.set_model(model_inst)

        if channel is None:
            self._stub = None
        else:
            self._stub = elasticdl_pb2_grpc.MasterStub(channel)
        self._max_minibatch_retry_num = max_minibatch_retry_num
        self._model_version = -1
        self._task_data_service = TaskDataService(
            self, self._job_type == JobType.TRAINING_WITH_EVALUATION
        )
        self._get_model_steps = get_model_steps

    # TODO: Multiple tests are currently using this function to initialize
    # self._model, where the initialization should be done via constructor.
    def set_model(self, model_inst):
        """Set model instance to worker."""
        self._model = model_inst
        self._train_eagerly = False
        self._init_embedding_layer()
        self._var_created = self._model.built
        self._non_embed_vars = []
        if self._var_created:
            self._non_embed_vars = get_non_embedding_trainable_vars(
                self._model, self._embedding_layers
            )

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
        # The first `non_embed_vars_n` items in `grads` are gradients for
        # `self._non_embed_vars`
        for g, v in zip(grads[:non_embed_vars_n], self._non_embed_vars):
            if isinstance(g, tf.IndexedSlices):
                req.gradient[v.name].CopyFrom(
                    ndarray_to_tensor(
                        g.values.numpy(), tuple(g.indices.numpy())
                    )
                )
            else:
                req.gradient[v.name].CopyFrom(ndarray_to_tensor(g.numpy()))

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
                bet_number += len(layer.bet_ids_pair)
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
                for _, ids in layer.bet_ids_pair:
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
        self._non_embed_vars = get_non_embedding_trainable_vars(
            self._model, self._embedding_layers
        )

        if not self._var_created:
            self.report_variable()
            self._var_created = True

        if self._need_embedding_layer_check:
            self._train_eagerly = False
            for layer in self._embedding_layers:
                if len(layer.bet_ids_pair) > 1:
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
                bets.extend([batch_embedding for (batch_embedding, _) in layer.bet_ids_pair])
        return self._non_embed_vars + bets

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
