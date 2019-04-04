import tensorflow as tf
from tensorflow.python.ops import math_ops
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

    def get_model(self):
        # TODO: get model from master
        pass

    def report_task_result(self):
        # TODO: report task result to master
        pass

    def report_gradient(self):
        # TODO: report gradient to ps
        pass

    def distributed_train(self):
        # TODO: distributed training
        pass

    def local_train(self, batch_size, epoch=1, kwargs=None):
        """
        Local training for local testing. Must in eager mode.
        Argments:
            batch_size: batch size in training
            epoch: the number of epoch in training
            kwargs: contains a dict of parameters used in training
        """

        if not tf.executing_eagerly():
            raise ValueError('Eager mode is required for LocalTrain')

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
