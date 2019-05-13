import tensorflow as tf
assert tf.executing_eagerly()

from google.protobuf import empty_pb2
from tensorflow.python.ops import math_ops
from elasticdl.proto import master_pb2_grpc
from elasticdl.proto import master_pb2
from elasticdl.common.ndarray import ndarray_to_tensor, tensor_to_ndarray
from elasticdl.common.model_helper import load_user_model
import itertools
import recordio

# the default max number of a minibatch retrain as its gradients are not accepted by master.
DEFAULT_MAX_MINIBATCH_RETRAIN_NUM = 64

class Worker(object):
    """ElasticDL worker"""

    def __init__(self,
                 model_file,
                 channel=None,
                 max_retrain_num=DEFAULT_MAX_MINIBATCH_RETRAIN_NUM):
        """
        Arguments:
            model_module: A module to define the model
            channel: grpc channel
            max_retrain_num: max number of a minibatch retrain as its gradients are not accepted by master
        """

        model_module = load_user_model(model_file)
        self._model = model_module.model
        self._input_fn = model_module.input_fn 
        self._opt_fn = model_module.optimizer
        self._loss = model_module.loss
        self._input_names = model_module.input_names

        if channel is None:
            self._stub = None
        else:
            self._stub = master_pb2_grpc.MasterStub(channel)
        self._max_retrain_num = max_retrain_num
        self._model_version = -1

    def get_task(self):
        """
        get task from master
        """
        return self._stub.GetTask(empty_pb2.Empty())

    def get_model(self, min_version):
        """
        get model from master, and update model_version
        """
        req = master_pb2.GetModelRequest()
        req.min_version = min_version
        model = self._stub.GetModel(req)

        for var in self._model.trainable_variables:
            # Assumes all trainable variables exist in model.param.
            var.assign(
                tensor_to_ndarray(model.param[var.name]))
        self._model_version = model.version

    def report_task_result(self, task_id, err_msg):
        """
        report task result to master
        """
        report = master_pb2.ReportTaskResultRequest()
        report.task_id = task_id
        report.err_message = err_msg
        return self._stub.ReportTaskResult(report)

    def report_gradient(self, grads):
        """
        report gradient to ps, return (accepted, model_version) from rpc call.
        """
        req = master_pb2.ReportGradientRequest()
        for g, v in zip(grads, self._model.trainable_variables):
            req.gradient[v.name].CopyFrom(
                ndarray_to_tensor(g.numpy()))
        req.model_version = self._model_version
        res = self._stub.ReportGradient(req)
        return res.accepted, res.model_version

    def distributed_train(self):
        """
        Distributed training.
        """
        while True:
            task = self.get_task()
            if not task.shard_file_name:
                # No more task
                break
            batch_size = task.minibatch_size
            err_msg = ""
            try:
                with recordio.File(task.shard_file_name, "r") as rdio_r:
                    reader = rdio_r.get_reader(task.start, task.end)
                    min_model_version = task.model_version
                    while True:
                        record_buf = list(
                            itertools.islice(reader, 0, batch_size))
                        if not record_buf:
                            break

                        for _ in range(self._max_retrain_num):
                            # TODO: optimize the logic to avoid unnecessary get_model call.
                            self.get_model(
                                max(self._model_version, min_model_version))

                            batch_input_data, batch_label = self._input_fn(record_buf)

                            with tf.GradientTape() as tape:
                                inputs = []
                                for input_name in self._input_names:
                                    inputs.append(batch_input_data[input_name])
                                if len(inputs) == 1:
                                    inputs = inputs[0]
                                outputs = self._model.call(inputs, training=True)
                                loss = self._loss(outputs, batch_label)

                                # TODO:  Add regularization loss if any,
                                #        which should be divided by the number of contributing workers.
                            grads = tape.gradient(
                                loss, self._model.trainable_variables)
                            print("Loss is ", loss.numpy())

                            accepted, min_model_version = self.report_gradient(
                                grads)
                            if accepted:
                                break
                        else:
                            # Worker got stuck, fail the task.
                            # TODO: stop the worker if it fails to make any progress for some time.
                            raise RuntimeError("Worker got stuck")


            except Exception as ex:
                err_msg = str(ex)
            self.report_task_result(task.task_id, err_msg)

    def local_train(self, file_list, batch_size, epoch=1, kwargs=None):
        """
        Local training for local testing. Must in eager mode.
        Argments:
            batch_size: batch size in training
            epoch: the number of epoch in training
            kwargs: contains a dict of parameters used in training
        """
        optimizer = self._opt_fn()
        for _ in range(epoch):
            for f in file_list:
                with recordio.File(f, "r") as rdio_r:
                    reader = rdio_r.get_reader(0, rdio_r.count())
                    while True:
                        record_buf = list(
                            itertools.islice(reader, 0, batch_size))
                        if not record_buf:
                            break

                        data, labels = self._input_fn(record_buf)

                        with tf.GradientTape() as tape:
                            inputs = []
                            for input_name in self._input_names:
                                inputs.append(data[input_name])
                            if len(inputs) == 1:
                                inputs = inputs[0]
                            outputs = self._model.call(inputs, training=True)
                            loss = self._loss(outputs, labels)

                            # Add regularization loss if any.
                            # Note: for distributed training, the regularization loss should
                            #       be divided by the number of contributing workers, which
                            #       might be difficult for elasticdl.
                            if self._model.losses:
                                loss += math_ops.add_n(self._model.losses)
                        grads = tape.gradient(
                            loss, self._model.trainable_variables)
                        optimizer.apply_gradients(
                            zip(grads, self._model.trainable_variables))
                        print("Loss is ", loss.numpy())
