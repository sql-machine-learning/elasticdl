import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils


class Embedding(tf.keras.layers.Layer):
    """
    Input: indexes for the embedding entries
           shape is (batch_size, input_length)
    Output:
      corresponding (combined) embeddings with a shape of
      (batch_size, input_length, output_dim) if combiner is None
      (batch_size, output_dim) if combiner is not None
    Arguments:
      output_dim: the dimension of the embedding vector
      embedding_initializer: Initializer for embedding table
      mask_zero: Whether or not the input value 0 is a special "padding"
        value that should be masked out.
      input_length: Length of input sequences, when it is constant.
        This argument is required if you are going to connect
        `Flatten` then `Dense` layers upstream
        (without it, the shape of the dense outputs cannot be computed).
      combiner: A string specifying the reduction op or None if not used.
        "mean", "sqrtn" and "sum" are supported for the reduction op.
    TODO: support mask_zero
    TODO: support combiner
    TODO: support sparse input
    """

    def __init__(
        self,
        output_dim,
        embedding_initializer="uniform",
        mask_zero=False,
        input_length=None,
        combiner=None,
        **kwargs
    ):
        if "input_shape" not in kwargs and input_length:
            kwargs["input_shape"] = (input_length,)
        super(Embedding, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.embedding_initializer = embedding_initializer
        self.mask_zero = mask_zero
        self.input_length = input_length
        self.combiner = combiner
        self.tape = None
        self.worker = None
        self.bet_ids_pair = []

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
        batch_embedding = self.worker.embedding_lookup(
            unique_ids, self._name, self.embedding_initializer
        )
        return batch_embedding

    def call(self, input):
        ids = tf.convert_to_tensor(input, name="embedding_ids")
        flat_ids = tf.reshape(ids, [-1])
        unique_ids, idx = tf.unique(flat_ids)
        batch_embedding_tensor = tf.py_function(
            self.lookup_embedding, inp=[unique_ids], Tout=tf.float32
        )
        if self.tape:
            # tape.watch works with eager mode only
            if not tf.executing_eagerly():
                raise RuntimeError("tape.watch only works with eager mode")
            self.tape.watch(batch_embedding_tensor)
            self.bet_ids_pair.append((batch_embedding_tensor, unique_ids))
        outputs = tf.gather(batch_embedding_tensor, idx)
        outputs = tf.reshape(
            outputs, ids.get_shape().concatenate(self.output_dim)
        )
        return outputs

    def reset(self):
        self.bet_ids_pair = []
        self.tape = None

    def set_tape(self, tape):
        self.tape = tape

    def set_worker(self, worker):
        self.worker = worker
