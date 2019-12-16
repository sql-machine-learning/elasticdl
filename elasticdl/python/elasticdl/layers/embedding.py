import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import tf_utils

from elasticdl.python.elasticdl.embedding_delegate import EmbeddingAndIds


class Embedding(tf.keras.layers.Layer):
    """
    Input: indexes for the embedding entries with a shape of
      (batch_size, input_length). Input can be either dense tensor
      or SparseTensor.
    Output:
      corresponding (combined) embeddings with a shape of
      (batch_size, input_length, output_dim) if combiner is None
      (batch_size, output_dim) if combiner is not None
    Arguments:
      output_dim: the dimension of the embedding vector
      input_dim: the max input id. If 0 or None, will not check the range of
        input embedding ids.
      embeddings_initializer: Initializer for embedding table
      mask_zero: Whether or not the input value 0 is a special "padding"
        value that should be masked out.
        If input is SparseTensor, mask_zero must be False.
      input_length: Length of input sequences, when it is constant.
        This argument is required if you are going to connect
        `Flatten` then `Dense` layers upstream
        (without it, the shape of the dense outputs cannot be computed).
      combiner: A string specifying the reduction op or None if not used.
        "mean", "sqrtn" and "sum" are supported for the reduction op.
        If input is SparseTensor, combiner must set as a reduction op.
    """

    def __init__(
        self,
        output_dim,
        input_dim=None,
        embeddings_initializer="uniform",
        mask_zero=False,
        input_length=None,
        combiner=None,
        **kwargs
    ):
        if "input_shape" not in kwargs and input_length:
            kwargs["input_shape"] = (input_length,)
        super(Embedding, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero
        self.input_length = input_length
        self.combiner = combiner
        self.tape = None
        self._lookup_embedding_func = None

        self._embedding_and_ids_eagerly = []

        # BET's shape and ids' shape in `self._embedding_and_ids_graph` have
        # `None` dimension. This is because they have different shapes in
        # different iterations.
        # `tf.Variable` requires initial value if shape has `None` dimension.
        self._embedding_and_ids_graph = []

    def _init_for_graph_mode(self):
        self._embedding_and_ids_graph = [
            EmbeddingAndIds(
                batch_embedding=tf.Variable(
                    # In some cases, `tf.Variable` requires that initial value
                    # is callable.
                    initial_value=lambda: tf.zeros((1, self.output_dim)),
                    shape=tf.TensorShape((None, self.output_dim)),
                    dtype=tf.float32,
                    trainable=True,
                ),
                batch_ids=tf.Variable(
                    initial_value=lambda: tf.zeros((1, 1), dtype=tf.int64),
                    shape=tf.TensorShape(None),
                    dtype=tf.int64,
                    trainable=False,
                ),
            )
        ]

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        # this function is taken from
        # tf.keras.layers.Embedding.compute_output_shape
        # https://github.com/tensorflow/tensorflow/blob/3f3c728bf80e0fd6653744318cbbfe1454c6ddca/tensorflow/python/keras/layers/embeddings.py#L156
        if self.input_length is None:
            return input_shape + (self.output_dim,)
        else:
            if isinstance(self.input_length, (list, tuple)):
                in_lens = list(self.input_length)
            else:
                in_lens = [self.input_length]
            if len(in_lens) != len(input_shape) - 1:
                raise ValueError(
                    '"input_length" is %s, '
                    "but received input has shape %s"
                    % (str(self.input_length), str(input_shape))
                )
            else:
                for i, (s1, s2) in enumerate(zip(in_lens, input_shape[1:])):
                    if s1 is not None and s2 is not None and s1 != s2:
                        raise ValueError(
                            '"input_length" is %s, '
                            "but received input has shape %s"
                            % (str(self.input_length), str(input_shape))
                        )
                    elif s1 is None:
                        in_lens[i] = s2
            return (input_shape[0],) + tuple(in_lens) + (self.output_dim,)

    @property
    def name(self):
        return self._name

    @staticmethod
    def get_key(name_list):
        return "-".join(map(str, name_list))

    def lookup_embedding(self, unique_ids):
        ids = unique_ids.numpy()
        self._check_id_valid(ids)
        if self._lookup_embedding_func:
            embedding_vectors = self._lookup_embedding_func(self._name, ids)
            return embedding_vectors

    def _check_id_valid(self, ids):
        if not self.input_dim:
            return
        first_may_exceed_id = ids[np.argmax(ids >= self.input_dim)]
        if self.input_dim is not None and first_may_exceed_id > self.input_dim:
            raise ValueError(
                " The embedding id cannot be bigger "
                "than input_dim. id = %d is not in [0, %d)"
                % (first_may_exceed_id, self.input_dim)
            )
        if any(ids < 0):
            raise ValueError(
                "The embedding id cannot be less than zero. "
                "Found invalid id %d" % ids[ids < 0][0]
            )

    def _record_gradients(self, batch_embedding, ids):
        if tf.executing_eagerly():
            self.tape.watch(batch_embedding)
            self._embedding_and_ids_eagerly.append(
                EmbeddingAndIds(batch_embedding, ids)
            )
        else:
            # In graph mode, assigning tensors to trainable variables is
            # allowed and tape can record the gradients of trainable
            # variables automatically.
            embedding_and_ids = self._embedding_and_ids_graph[0]
            embedding_and_ids.batch_embedding.assign(batch_embedding)
            embedding_and_ids.batch_ids.assign(ids)
            batch_embedding = embedding_and_ids.batch_embedding
        return batch_embedding

    def call(self, input):
        if (
            self.tape
            and not tf.executing_eagerly()
            and not self._embedding_and_ids_graph
        ):
            self._init_for_graph_mode()

        input = tf.cast(input, tf.int64)
        if isinstance(input, tf.SparseTensor):
            return self._sparse_input_call(input)

        ids = tf.convert_to_tensor(input, name="embedding_ids")
        flat_ids = tf.reshape(ids, [-1])
        unique_ids, idx = tf.unique(flat_ids)

        # There is a memory leak when using tf.py_function with eager
        # execution. So, we only use tf.py_function in graph mode.
        if isinstance(unique_ids, ops.EagerTensor):
            batch_embedding = self.lookup_embedding(unique_ids)
            batch_embedding = tf.constant(batch_embedding)
        else:
            batch_embedding = tf.py_function(
                self.lookup_embedding, inp=[unique_ids], Tout=tf.float32
            )

        # Gradient for `batch_embedding` is SparseTensor here due to
        # `tf.gather` op. `tf.gather` accesses tensor slices, resulting in
        # sparse tensor gradient.
        # TODO: use tf.cond rather than python if statement
        if self.tape:
            batch_embedding = self._record_gradients(batch_embedding, flat_ids)

        outputs = tf.gather(batch_embedding, idx)
        # tf.reshape does not support shape with None. Replace None with -1.
        if ids.get_shape().rank == 2:
            input_length = ids.get_shape()[1]
            if input_length is None:
                outputs.set_shape(shape=(None, None, self.output_dim))
                return outputs
            output_shape = (-1, input_length, self.output_dim)
        else:
            output_shape = ids.get_shape().concatenate(self.output_dim)
        outputs = tf.reshape(outputs, output_shape)
        # TODO: support combiner for dense input
        return outputs

    def _sparse_input_call(self, sparse_input):
        if self.combiner not in ["sum", "mean", "sqrtn"]:
            raise ValueError(
                "combiner must set sum, mean or sqrtn for sparse input"
            )
        unique_ids, idx = tf.unique(sparse_input.values)
        # Gradient for `batch_embedding` is dense tensor.
        batch_embedding = tf.py_function(
            self.lookup_embedding, inp=[unique_ids], Tout=tf.float32
        )
        # TODO: use tf.cond rather than python if statement
        if self.tape:
            batch_embedding = self._record_gradients(
                batch_embedding, unique_ids
            )

        segment_ids = sparse_input.indices[:, 0]
        if segment_ids.dtype != tf.int32:
            segment_ids = tf.cast(segment_ids, tf.int32)

        if self.combiner == "sum":
            batch_embedding = tf.sparse.segment_sum(
                batch_embedding, idx, segment_ids
            )
        elif self.combiner == "mean":
            batch_embedding = tf.sparse.segment_mean(
                batch_embedding, idx, segment_ids
            )
        elif self.combiner == "sqrtn":
            batch_embedding = tf.sparse.segment_sqrt_n(
                batch_embedding, idx, segment_ids
            )
        batch_embedding.set_shape((None, self.output_dim))
        return batch_embedding

    def compute_mask(self, inputs, mask=None):
        if isinstance(input, tf.SparseTensor):
            raise ValueError("SparseTensor inputs do not support mask_zero")
        if not self.supports_masking:
            return None
        return tf.math.not_equal(inputs, 0)

    def reset(self):
        self._embedding_and_ids_eagerly = []
        self.tape = None

    def set_tape(self, tape):
        self.tape = tape

    def set_lookup_embedding_func(self, func):
        """Sets function for looking up embeddings in the PS.

        Args:
            func: The function used to look up embeddings. The arguments of
                are `(layer_name, embedding_id_list)`, where `layer_name` is
                the name of embedding layer, and `embedding_id_list` is a list
                of embedding ids to be looked up.
        """
        self._lookup_embedding_func = func

    @property
    def embedding_and_ids(self):
        """
        Return bet and ids pairs.
        """
        if self._embedding_and_ids_eagerly:
            return self._embedding_and_ids_eagerly
        return self._embedding_and_ids_graph
