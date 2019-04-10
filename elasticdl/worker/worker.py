import tensorflow as tf
assert tf.executing_eagerly()

from tensorflow.python.ops import math_ops
import recordio
from proto import master_pb2
from proto import master_pb2_grpc

class Worker(object):
    """ElasticDL worker"""

    def __init__(self, model_cls, input_fn, opt_fn, channel=None):
        """
        Arguments:
            model_cls: A class to define the model, which contains funcs
                get_keras_model: return the keras model defined in the class, with a tf dataset as its input
                output(data): get model ouput from data as input, either a single output of a dict of outputs
                loss(output, data): get model loss from output and data as input
            input_fn: a func to to get a dataset, which can be used as the keras model input
                      dataset = input_fn(dict_of_params)
                      dict_of_params from GetTask for DistributedTrain, from kwargs for LocalTrain
            opt_fn: a func to get the optimizer 
            channel: grpc channel
        """

        self._model_cls = model_cls()
        self._keras_model = self._model_cls.get_keras_model()
        self._input_fn = input_fn
        self._opt_fn = opt_fn
        if channel is None:
            self._stub = None
        else:
            self._stub = master_pb2_grpc.MasterStub(channel)
        self._model_version = -1

    def get_task(self):
        # TODO: get task from master
        pass

    def get_model(self, min_version):
        # TODO: get model from master, and update model_version
        pass

    def report_task_result(self, task_id, err_msg):
        # TODO: report task result to master
        pass

    def report_gradient(self, grads, variables):
        # TODO: report gradient to ps
        pass

    def distributed_train(self):
        """
        Distributed training.
        """
        while True:
            task = self.get_task()
            if task.shard_file_name is '':
                # No more task
                break
            batch_size = task.minibatch_size
            task_data_size = task.end - task.start + 1
            record_buf = []
            err_msg = ''
            record_count = 0
            try:
                with recordio.File(task.shard_file_name, 'r') as rdio_r:
                    for record in rdio_r.get_reader(task.start, task.end + 1):
                        record_count += 1
                        record_buf.append(record)
                        if len(record_buf) == batch_size or record_count == task_data_size:
                            # TODO: optimize the logic to avoid unnecessary get_model call.
                            self.get_model(
                                max(self._model_version, task.model_version))

                            batch_input_data = self._input_fn(record_buf)

                            with tf.GradientTape() as tape:
                                output = self._model_cls.output(
                                    batch_input_data)
                                loss = self._model_cls.loss(
                                    output, batch_input_data)
                                # TODO:  Add regularization loss if any,
                                #        which should be divided by the number of contributing workers.
                            grads = tape.gradient(
                                loss, self._keras_model.variables)
                            print('Loss is ', loss.numpy())

                            self.report_gradient(
                                grads, self._keras_model.variables)
                            record_buf = []
            except Exception as ex:
                err_msg = str(ex)
            self.report_task_result(task.task_id, err_msg)

    def local_train(self, batch_size, epoch=1, kwargs=None):
        """
        Local training for local testing. Must in eager mode.
        Argments:
            batch_size: batch size in training
            epoch: the number of epoch in training
            kwargs: contains a dict of parameters used in training
        """

        dataset = self._input_fn(kwargs)
        dataset = dataset.repeat(epoch).batch(batch_size)
        optimizer = self._opt_fn()

        for data in dataset:
            with tf.GradientTape() as tape:
                output = self._model_cls.output(data)
                loss = self._model_cls.loss(output, data)
                # Add regularization loss if any.
                # Note: for distributed training, the regularization loss should
                #       be divided by the number of contributing workers, which
                #       might be difficult for elasticdl.
                if self._keras_model.losses:
                    loss += math_ops.add_n(self._keras_model.losses)
            grads = tape.gradient(loss, self._keras_model.variables)
            optimizer.apply_gradients(zip(grads, self._keras_model.variables))
            print('Loss is ', loss.numpy())
        pass
