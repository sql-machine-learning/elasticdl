import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils

from elasticdl.python.elasticdl.embedding_delegate import EmbeddingDelegate


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
        self._embedding_delegate = EmbeddingDelegate(
            input_dim=input_dim, output_dim=output_dim, name=self.name
        )

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
        return self._embedding_delegate.lookup_embedding(unique_ids)

    def call(self, input):
        if self.tape:
            self._embedding_delegate.init_for_graph_mode_if_necessary()

        input = tf.cast(input, tf.int64)
        if isinstance(input, tf.SparseTensor):
            return self._sparse_input_call(input)

        ids = tf.convert_to_tensor(input, name="embedding_ids")
        flat_ids = tf.reshape(ids, [-1])
        unique_ids, idx = tf.unique(flat_ids)
        # Gradient for `batch_embedding` is SparseTensor here due to
        # `tf.gather` op. `tf.gather` accesses tensor slices, resulting in
        # sparse tensor gradient.
        batch_embedding = tf.py_function(
            self.lookup_embedding, inp=[unique_ids], Tout=tf.float32
        )
        # TODO: use tf.cond rather than python if statement
        if self.tape:
            batch_embedding = self._embedding_delegate.record_gradients(
                self.tape, batch_embedding, flat_ids
            )

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
            batch_embedding = self._embedding_delegate.record_gradients(
                self.tape, batch_embedding, unique_ids
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
        self.tape = None
        self._embedding_delegate.reset()

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
        self._embedding_delegate.set_lookup_embedding_func(func)

    @property
    def embedding_and_ids(self):
        """
        Return bet and ids pairs.
        """
        return self._embedding_delegate.embedding_and_ids
