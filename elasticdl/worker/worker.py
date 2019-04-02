import tensorflow as tf
from proto import master_pb2
from proto import master_pb2_grpc


class Worker(object):
    """ElasticDL worker"""

    def __init__(self, model_cls, input_fn, opt_fn, channel=None):
        """
        Arguments:
            model_cls: A class to define the model, which contains funcs
                GetKerasModel: return the keras model defined in the class, with a tf dataset as its input
                Output(data): get model ouput from data as input, either a single output of a dict of outputs
                Loss(data): get model loss from data as input
            input_fn: a func to to get a dataset, which can be used as the keras model input
                      dataset = input_fn(dict_of_params)
                      dict_of_params from GetTask for DistributedTrain, from kwargs for LocalTrain
            opt_fn: a func to get the optimizer 
            channel: grpc channel
        """

        self._model_cls = model_cls()
        self._keras_model = self._model_cls.GetKerasModel()
        self._input_fn = input_fn
        self._opt_fn = opt_fn
        if channel is None:
            self._stub = None
        else:
            self._stub = master_pb2_grpc.MasterStub(channel)
        self._model_version = -1

    def GetTask(self):
        # TODO: get task from master
        pass

    def GetModel(self):
        # TODO: get model from master
        pass

    def ReportTaskResult(self):
        # TODO: report task result to master
        pass

    def ReportGradient(self):
        # TODO: report gradient to ps
        pass

    def DistributedTrain(self):
        # TODO: distributed training
        pass

    def LocalTrain(self, batch_size, epoch=1, kwargs=None):
        """
        Local training for local testing. Must in eager mode.
        Argments:
            epoch: the number of epoch in training
            kwargs: contains a dict of parameters used in training
        """

        if not tf.executing_eagerly():
            print('Eager mode is required for LocalTrain')
            return

        dataset = self._input_fn(kwargs)
        dataset = dataset.repeat(epoch).batch(batch_size)
        optimizer = self._opt_fn()

        for data in dataset:
            with tf.GradientTape() as tape:
                output = self._model_cls.Output(data)
                loss = self._model_cls.Loss(output, data)
            grads = tape.gradient(loss, self._keras_model.variables)
            optimizer.apply_gradients(zip(grads, self._keras_model.variables))
            print('Loss is ', loss.numpy())
        pass
