import logging
import traceback

import tensorflow as tf

assert tf.executing_eagerly()  # noqa

import recordio

from contextlib import closing
from elasticdl.proto import elasticdl_pb2_grpc
from elasticdl.proto import elasticdl_pb2
from elasticdl.python.elasticdl.common.ndarray import (
    ndarray_to_tensor,
    tensor_to_ndarray,
)
from elasticdl.python.elasticdl.common.model_helper import load_module

# The default maximum number of a minibatch retry as its results
# (e.g. gradients) are not accepted by master.
DEFAULT_MAX_MINIBATCH_RETRY_NUM = 64


class Worker(object):
    """ElasticDL worker"""

    def __init__(
        self,
        worker_id,
        model_file,
        channel=None,
        max_minibatch_retry_num=DEFAULT_MAX_MINIBATCH_RETRY_NUM,
        codec_file=None,
    ):
        """
        Arguments:
            model_file: A module to define the model
            channel: grpc channel
            max_minibatch_retry_num: The maximum number of a minibatch retry
                as its results (e.g. gradients) are not accepted by master.
        """
        self._logger = logging.getLogger(__name__)
        self._worker_id = worker_id
        model_module = load_module(model_file)
        self._model = model_module.model
        self._feature_columns = model_module.feature_columns()
        self._var_created = self._model.built
        self._input_fn = model_module.input_fn
        self._opt_fn = model_module.optimizer
        self._loss = model_module.loss
        self._eval_metrics_fn = model_module.eval_metrics_fn
        all_columns = self._feature_columns + model_module.label_columns()

        # Initilize codec
        codec_module = load_module(codec_file)
        codec_module.codec.init(all_columns)
        self._codec = codec_module.codec

        if channel is None:
            self._stub = None
        else:
            self._stub = elasticdl_pb2_grpc.MasterStub(channel)
        self._max_minibatch_retry_num = max_minibatch_retry_num
        self._model_version = -1

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

    @staticmethod
    def _get_batch(reader, batch_size):
        res = []
        for i in range(batch_size):
            record = reader.record()
            if record is None:
                break
            res.append(record)
            # res.append(decode(record))
        return res

    # def _get_features_and_labels(self, record_buf):
    #     batch_input_data, batch_labels = self._input_fn(record_buf)
    #     features = [
    #         batch_input_data[f_col.key] for f_col in self._feature_columns
    #     ]
    #     if len(features) == 1:
    #         features = features[0]
    #     return features, batch_labels

    def _create_variable_and_report(self, features):
        # Use model.call to create variables, then report to ps
        _ = self._model.call(features)
        self.report_variable()
        self._var_created = True

    def _run_training_task(self, features, labels):
        with tf.GradientTape() as tape:
            outputs = self._model.call(features, training=True)
            loss = self._loss(outputs, labels)

            # TODO:  Add regularization loss if any,
            #        which should be divided by the
            #        number of contributing workers.
        grads = tape.gradient(loss, self._model.trainable_variables)
        accepted, min_model_version = self.report_gradient(grads)
        return accepted, min_model_version, loss

    def _run_evaluation_task(self, features, labels):
        outputs = self._model.call(features, training=False)
        evaluation_metrics = self._eval_metrics_fn(outputs, labels)
        return self.report_evaluation_metrics(evaluation_metrics)

    def _handle_task(self, task):
        min_model_version = task.model_version
        with closing(
            recordio.Scanner(
                task.shard_file_name, task.start, task.end - task.start
            )
        ) as reader:
            while True:
                record_buf = self._get_batch(reader, task.minibatch_size)
                if not record_buf:
                    break
                min_model_version = self._process_minibatch(
                    task, record_buf, min_model_version
                )

    def _process_minibatch(self, task, record_buf, min_model_version):
        feature_tensor_list, label_nparray = self._input_fn(record_buf, self._codec.decode)
        # zjl?: why?
        if len(feature_tensor_list) == 1:
            features = feature_tensor_list[0]
        # features, labels = self._get_features_and_labels(record_buf)

        if not self._var_created:
            self._create_variable_and_report(features)
        for _ in range(self._max_minibatch_retry_num):
            if task.type == elasticdl_pb2.EVALUATION:
                self.get_model(min_model_version, elasticdl_pb2.FIXED)
                accepted, _ = self._run_evaluation_task(features, label_nparray)
                if accepted:
                    break
            elif task.type == elasticdl_pb2.TRAINING:
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
                    self._logger.info("Loss is %f" % loss.numpy())
                    break
            else:
                raise RuntimeError("Unrecognized task type, %s" % task.type)
        else:
            # Worker got stuck, fail the task.
            # TODO: stop the worker if it fails to make any
            #       progress for some time.
            raise RuntimeError("Worker got stuck")
        return min_model_version

    def run(self):
        """
        Fetches task from master and performs training or evaluation.
        """
        while True:
            task = self.get_task()
            if not task.shard_file_name:
                # No more task
                break
            err_msg = ""
            try:
                self._handle_task(task)
            except RuntimeError as err:
                err_msg = str(err)
                traceback.print_exc()
            except Exception as ex:
                err_msg = str(ex)
                traceback.print_exc()
                raise ex
            self.report_task_result(task.task_id, err_msg)
