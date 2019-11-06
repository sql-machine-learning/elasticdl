import tensorflow as tf

from elasticdl.python.common.tensor import tensor_pb_to_ndarray
from elasticdl.python.ps.embedding_table import create_embedding_table


class Parameters(object):
    """
    There are two kinds of parameters:

    1. non-embedding parameters, or dense tensor parameters. We save it
       in a hashmap `non-embedding_params`, the key is the parameter name,
       the value is a tf.Variable` object.
    2. embedding parameters, or row-sparse parameters. We save it in a
       hashmap `embedding_params`, the key is the embedding layer name,
       the value is an `EmbeddingTable` object.

    """

    def __init__(self):
        self.version = 0
        self.init_status = False
        self.non_embedding_params = {}
        self.embedding_params = {}

    def get_non_embedding_param(self, name, default_value=None):
        return self.non_embedding_params.get(name, default_value)

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

    def check_grad(self, grad):
        name = grad.name
        if name in self.non_embedding_params:
            param_shape = tuple(
                self.non_embedding_params[name].get_shape().as_list()
            )
            if grad.is_indexed_slices():
                dim0 = tf.math.reduce_max(grad.indices).numpy()
                dim1 = grad.values.shape[1]
                if dim0 > param_shape[0] or dim1 != param_shape[1]:
                    raise ValueError(
                        "Keras embedding param error: \
                        the shape of gradient %s is (%d, %d), \
                        the shape of parameter %s is (%d, %d), \
                        which is incompatible",
                        name,
                        dim0,
                        dim1,
                        name,
                        param_shape[0],
                        param_shape[1],
                    )
            else:
                if grad.values.shape != param_shape:
                    raise ValueError(
                        "Non embedding param error: \
                        the shape of gradient %s is %s, \
                        the shape of parameter %s is %s, \
                        which is incompatible",
                        name,
                        str(grad.values.shape),
                        name,
                        str(param_shape),
                    )
        elif name in self.embedding_params:
            if grad.values.shape[1] != self.embedding_params[name].dim:
                raise ValueError(
                    "ElasticDL embedding param error: \
                    the shape of gradient %s is (None, %d), \
                    the shape of parameter %s is (None, %d), \
                    which is incompatible",
                    name,
                    grad.values.shape[1],
                    name,
                    self.embedding_params[name].dim,
                )
        else:
            raise ValueError(
                "Name error: Gradient %s is not in Parameters", name
            )

    def init_from_model_pb(self, model_pb):
        if not self.init_status:
            tensors_pb = model_pb.param
            embeddings_pb = model_pb.embedding_table_info
            self._init_non_embedding_params(tensors_pb)
            self._init_embedding_params(embeddings_pb)
            self.version = model_pb.version
            self.init_status = True

    def _init_non_embedding_params(self, tensors_pb):
        for pb in tensors_pb:
            name = pb.name
            arr = tensor_pb_to_ndarray(pb)
            # This is hack here, `tf.Variable` has the magic! If you pass a
            # name "somename" to a `tf.Variable`, the final name will be
            # "somename:0". So, we have to truncate the input name first,
            # and wait for `tf.Variable` to add it back.
            var = tf.Variable(
                name=name[0:-2], initial_value=arr, trainable=True
            )
            self.non_embedding_params[var.name] = var

    def _init_embedding_params(self, embeddings_pb):
        for pb in embeddings_pb:
            self.embedding_params[pb.name] = create_embedding_table(pb)
