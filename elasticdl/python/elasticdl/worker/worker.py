import traceback
import tensorflow as tf
assert tf.executing_eagerly()

import itertools
import recordio

from contextlib import closing
from google.protobuf import empty_pb2
from tensorflow.python.ops import math_ops
from elasticdl.proto import master_pb2_grpc
from elasticdl.proto import master_pb2
from elasticdl.common.ndarray import ndarray_to_tensor, tensor_to_ndarray
from elasticdl.common.model_helper import load_user_model, build_model
from data.codec import TFExampleCodec
from data.codec import BytesCodec

# the default max number of a minibatch retrain as its gradients are not accepted by master.
DEFAULT_MAX_MINIBATCH_RETRAIN_NUM = 64

class Worker(object):
    """ElasticDL worker"""

    def __init__(self,
                 worker_id,
                 model_file,
                 channel=None,
                 max_retrain_num=DEFAULT_MAX_MINIBATCH_RETRAIN_NUM,
                 codec_type=None):
        """
        Arguments:
            model_module: A module to define the model
            channel: grpc channel
            max_retrain_num: max number of a minibatch retrain as its gradients are not accepted by master
        """
        self._worker_id = worker_id
        model_module = load_user_model(model_file)
        self._model = model_module.model
        self._feature_columns = model_module.feature_columns()
        build_model(self._model, self._feature_columns)
        self._input_fn = model_module.input_fn 
        self._opt_fn = model_module.optimizer
        self._loss = model_module.loss
        all_columns = self._feature_columns + model_module.label_columns()
        if codec_type == "tf_example":
            self._codec = TFExampleCodec(all_columns)
        elif codec_type == "bytes":
            self._codec = BytesCodec(all_columns)
        else:
            raise ValueError("invalid codec_type: " + codec_type)


        if channel is None:
            self._stub = None
        else:
            self._stub = master_pb2_grpc.MasterStub(channel)
        self._max_retrain_num = max_retrain_num
        self._model_version = -1
        self._codec_type = codec_type

    def get_task(self):
        """
        get task from master
        """
        req = master_pb2.GetTaskRequest()
        req.worker_id = self._worker_id
        
        return self._stub.GetTask(req)

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

    @staticmethod
    def _get_batch(reader, batch_size, decode):
        res = []
        for i in range(batch_size):
            record = reader.record()
            if record is None:
                break
            res.append(decode(record))
        return res
                

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
                with closing(recordio.Scanner(task.shard_file_name, task.start, task.end - task.start)) as reader:
                    min_model_version = task.model_version
                    while True:
                        record_buf = self._get_batch(reader, batch_size, self._codec.decode)
                        if not record_buf:
                            break

                        for _ in range(self._max_retrain_num):
                            # TODO: optimize the logic to avoid unnecessary get_model call.
                            self.get_model(
                                max(self._model_version, min_model_version))

                            batch_input_data, batch_label = self._input_fn(record_buf)

                            with tf.GradientTape() as tape:
                                inputs = []
                                for f_col in self._feature_columns:
                                    inputs.append(batch_input_data[f_col.key])
                                if len(inputs) == 1:
                                    inputs = inputs[0]
                                outputs = self._model.call(inputs, training=True)
                                loss = self._loss(outputs, batch_label.flatten())

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
                traceback.print_exc()
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
                with closing(recordio.Scanner(f)) as reader:
                    while True:
                        record_buf = self._get_batch(reader, batch_size, self._codec.decode)
                        if not record_buf:
                            break

                        data, labels = self._input_fn(record_buf)

                        with tf.GradientTape() as tape:
                            inputs = []
                            for f_col in self._feature_columns:
                                inputs.append(data[f_col.key])
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
