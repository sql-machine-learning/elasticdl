import tensorflow as tf

from elasticdl.python.common.ndarray import tensor_to_ndarray
from elasticdl.python.ps.embedding_table import create_embedding_table


class Parameters(object):
    def __init__(self):
        self.init_status = False
        self.non_embedding_params = {}
        self.embedding_params = {}

    def get_non_embedding_params(self):
        return self.non_embedding_params

    def set_non_embedding_params(self, variables):
        self.non_embedding_params = variables

    def get_embedding_param(self, name, indices):
        if name not in self.embedding_params:
            raise ValueError(
                "Please initialize embedding param %s first!", name
            )
        return self.embedding_params[name].get(indices)

    def set_embedding_param(self, name, indices, values):
        if name not in self.embedding_params:
            raise ValueError(
                "Please initialize embedding param %s first!", name
            )
        self.embedding_params[name].set(indices, values)

    def init_from_model_pb(self, model_pb):
        if not self.init_status:
            tensors_pb = model_pb.param
            embeddings_pb = model_pb.embedding_table_info
            self._init_non_embedding_params(tensors_pb)
            self._init_embedding_params(embeddings_pb)
            self.init_status = True

    def _init_non_embedding_params(self, tensors_pb):
        for pb in tensors_pb:
            name = pb.name
            arr = tensor_to_ndarray(pb)
            var = tf.Variable(name=name, initial_value=arr, trainable=True)
            self.non_embedding_params[name] = var

    def _init_embedding_params(self, embeddings_pb):
        for pb in embeddings_pb:
            table = create_embedding_table(pb)
            self.embedding_params[table.name] = table

    def clear(self):
        self.non_embedding_params.clear()
        self.embedding_params.clear()
