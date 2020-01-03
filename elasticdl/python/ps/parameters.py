import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.tensor import (
    Tensor,
    deserialize_tensor_pb,
    emplace_tensor_pb_from_ndarray,
    serialize_tensor,
    tensor_pb_to_ndarray,
)
from elasticdl.python.ps.embedding_table import (
    EmbeddingTable,
    create_embedding_table,
    get_slot_table_name,
)


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

    def reset(self):
        self.version = 0
        self.init_status = False
        self.non_embedding_params.clear()
        self.embedding_params.clear()

    def get_non_embedding_param(self, name, default_value=None):
        return self.non_embedding_params.get(name, default_value)

    def get_embedding_param(self, name, indices):
        if name not in self.embedding_params:
            raise ValueError(
                "Please initialize embedding param %s first!" % name
            )
        return self.embedding_params[name].get(indices)

    def set_embedding_param(self, name, indices, values):
        if name not in self.embedding_params:
            raise ValueError(
                "Please initialize embedding param %s first!" % name
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
                        "Keras embedding param error: "
                        "the shape of gradient %s is (%d, %d), "
                        "the shape of parameter %s is (%d, %d), "
                        "which is incompatible"
                        % (
                            name,
                            dim0,
                            dim1,
                            name,
                            param_shape[0],
                            param_shape[1],
                        )
                    )
            else:
                if grad.values.shape != param_shape:
                    raise ValueError(
                        "Non embedding param error: "
                        "the shape of gradient %s is %s, "
                        "the shape of parameter %s is %s, "
                        "which is incompatible"
                        % (
                            name,
                            str(grad.values.shape),
                            name,
                            str(param_shape),
                        )
                    )
        elif name in self.embedding_params:
            if grad.values.shape[1] != self.embedding_params[name].dim:
                raise ValueError(
                    "ElasticDL embedding param error: "
                    "the shape of gradient %s is (None, %d), "
                    "the shape of parameter %s is (None, %d), "
                    "which is incompatible"
                    % (
                        name,
                        grad.values.shape[1],
                        name,
                        self.embedding_params[name].dim,
                    )
                )
        else:
            raise ValueError(
                "Name error: Gradient %s is not in Parameters" % name
            )

    def init_from_model_pb(self, model_pb):
        """Initializes `Parameters` with model protocol buffer.

        The `Parameters` accepts model pb and initialize only when it is
        not initialized. Otherwise, it ignores the model pb.

        Args:
            model_pb: The model protocol buffer used for initialization.

        Returns:
            A bool indicates whether `Parameters` accepts this model pb or not.
        """
        if not self.init_status:
            tensors_pb = model_pb.param
            embeddings_pb = model_pb.embedding_table_info
            self.init_embedding_params(embeddings_pb)
            self._restore_params_from_pb(tensors_pb)
            self.version = model_pb.version
            self.init_status = True
            return True
        return False

    def _restore_params_from_pb(self, tensors_pb):
        for pb in tensors_pb:
            name = pb.name
            if not pb.indices:
                # Please note that `tf.Variable` will do something with magic.
                # If you pass a name "somename" to a `tf.Variable`, the final
                # variable name will be "somename:0". So the `tf.Variable.name`
                # is meaningless, we must avoid use it in PS side.
                arr = tensor_pb_to_ndarray(pb)
                var = tf.Variable(initial_value=arr, trainable=True)
                self.non_embedding_params[name] = var
            else:
                # Only pb of embedding parameters has indices.
                tensor = Tensor()
                deserialize_tensor_pb(pb, tensor)
                self.embedding_params[name].set(tensor.indices, tensor.values)

    def init_embedding_params(self, embeddings_pb):
        for pb in embeddings_pb:
            if pb.name not in self.embedding_params:
                self.embedding_params[pb.name] = create_embedding_table(pb)

    def has_embedding_params(self):
        return len(self.embedding_params) > 0

    def create_slot_params(self, slot_names, init_values):
        embed_layer_names = list(self.embedding_params.keys())
        for layer_name in embed_layer_names:
            for slot_name in slot_names:
                key = get_slot_table_name(layer_name, slot_name)
                if key in self.embedding_params:
                    raise ValueError(
                        "An embedding layer has unexpected name %s" % key
                    )
                self.embedding_params[key] = EmbeddingTable(
                    key,
                    self.embedding_params[layer_name].dim,
                    init_values[slot_name],
                    True,
                )

    def to_model_pb(self):
        """ Convert all parameters including embedding and non-embedding
        parameters to `elasticdl_pb2.Model` which can be serialized.
        """
        model_pb = elasticdl_pb2.Model()
        model_pb.version = self.version
        for name, var in self.non_embedding_params.items():
            emplace_tensor_pb_from_ndarray(
                model_pb.param, var.numpy(), name=name
            )

        for name, embedding_table in self.embedding_params.items():
            embedding_table_tensor = embedding_table.to_tensor()
            tensor_pb = model_pb.param.add()
            serialize_tensor(embedding_table_tensor, tensor_pb)

            embedding_info = embedding_table.to_embedding_table_info_pb()
            model_pb.embedding_table_info.append(embedding_info)

        return model_pb

    def debug_info(self):
        info = ""
        total_size = 0
        for param in self.embedding_params:
            info += self.embedding_params[param].debug_info()
            total_size += self.embedding_params[param].get_table_size()
        for param in self.non_embedding_params:
            shape = self.non_embedding_params[param].get_shape().as_list()
            size = (
                tf.size(self.non_embedding_params[param])
                * self.non_embedding_params[param].dtype.size
            )
            info += (
                "Non-embedding param name: %s\n  shape: %s\n  size: %d\n"
                % (param, str(shape), size)
            )
            total_size += size
        info += "Total parameters size: %d bytes" % total_size
        return info
