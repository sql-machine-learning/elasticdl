import traceback

import tensorflow as tf

from elasticdl.proto import elasticdl_pb2, elasticdl_pb2_grpc
from elasticdl.python.common.constants import JobType
from elasticdl.python.common.log_util import default_logger as logger
from elasticdl.python.common.model_helper import (
    load_model_from_module,
    load_module,
)
from elasticdl.python.common.ndarray import (
    ndarray_to_tensor,
    tensor_to_ndarray,
)
from elasticdl.python.worker.prediction_outputs_processor import (
    BasePredictionOutputsProcessor,
)
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
        model_file,
        dataset_fn="dataset_fn",
        loss="loss",
        optimizer="optimizer",
        eval_metrics_fn="eval_metrics_fn",
        channel=None,
        model_def=None,
        model_params="",
        prediction_outputs_processor="PredictionOutputsProcessor",
        max_minibatch_retry_num=DEFAULT_MAX_MINIBATCH_RETRY_NUM,
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
        model_module = load_module(model_file).__dict__
        self._model = load_model_from_module(
            model_def, model_module, model_params
        )
        self._var_created = self._model.built
        self._dataset_fn = model_module[dataset_fn]
        self._opt_fn = model_module[optimizer]
        self._loss = model_module[loss]
        self._eval_metrics_fn = model_module[eval_metrics_fn]

        if channel is None:
            self._stub = None
        else:
            self._stub = elasticdl_pb2_grpc.MasterStub(channel)
        self._max_minibatch_retry_num = max_minibatch_retry_num
        self._model_version = -1
        self._task_data_service = TaskDataService(
            self, self._job_type == JobType.TRAINING_WITH_EVALUATION
        )
        if prediction_outputs_processor in model_module:
            self._prediction_outputs_processor = model_module[
                prediction_outputs_processor
            ]()
            if not isinstance(
                self._prediction_outputs_processor,
                BasePredictionOutputsProcessor,
            ):
                logger.warning(
                    "prediction_outputs_processor is not "
                    "inherited from BasePredictionOutputsProcessor. "
                    "Prediction outputs may not be processed correctly."
                )
        else:
            self._prediction_outputs_processor = None

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

        for var in self._model.trainable_variables:
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
        for v in self._model.trainable_variables:
            req.variable[v.name].CopyFrom(ndarray_to_tensor(v.numpy()))
        self._stub.ReportVariable(req)

    def report_gradient(self, grads):
        """
        report gradient to ps, return (accepted, model_version) from rpc call.
        """
        req = elasticdl_pb2.ReportGradientRequest()
        for g, v in zip(grads, self._model.trainable_variables):
            req.gradient[v.name].CopyFrom(ndarray_to_tensor(g.numpy()))
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

    @tf.function
    def training_process(self, features, labels):
        with tf.GradientTape() as tape:
            outputs = self._model.call(features, training=True)
            loss = self._loss(outputs, labels)
            # Add regularization loss if any
            if self._model.losses:
                loss += tf.math.add_n(self._model.losses)
        grads = tape.gradient(loss, self._model.trainable_variables)
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
        return accepted, min_model_version, loss

    def _run_evaluation_task(self, features, labels):
        evaluation_metrics = self.evaluation_process(features, labels)
        return self.report_evaluation_metrics(evaluation_metrics)

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
                accepted, _ = self._run_evaluation_task(features, labels)
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
        eval_dataset = eval_info[0]
        model_version = eval_info[1]
        task_id = eval_info[2]
        eval_dataset = self._dataset_fn(eval_dataset, training=False)
        eval_dataset = eval_dataset.batch(self._minibatch_size).prefetch(1)
        err_msg = ""
        for data in eval_dataset:
            data_err_msg = self._process_minibatch_and_report(
                data[0], data[1], elasticdl_pb2.EVALUATION, model_version
            )
            if data_err_msg:
                err_msg = data_err_msg
                break
        del eval_dataset
        self.report_task_result(task_id, err_msg)

    def _process_minibatch_and_report(
        self, features, labels, task_type, model_version
    ):
        err_msg = ""
        try:
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
        job_is_training = (
            self._job_type == JobType.TRAINING_ONLY
            or self._job_type == JobType.TRAINING_WITH_EVALUATION
        )
        while True:
            dataset = self._task_data_service.get_dataset()
            if not dataset:
                break
            dataset = self._dataset_fn(dataset, training=job_is_training)
            dataset = dataset.batch(self._minibatch_size).prefetch(1)
            for d in dataset:
                if self._job_type == JobType.TRAINING_WITH_EVALUATION:
                    self._process_eval_task_if_needed()
                task = self._task_data_service.get_current_task()
                err_msg = self._process_minibatch_and_report(
                    d[0], d[1], task.type, task.model_version
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
