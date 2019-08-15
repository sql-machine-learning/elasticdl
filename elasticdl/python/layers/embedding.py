import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils


class EdlEmbedding(tf.keras.layers.Layer):
    """
    Input: indexes for the embedding entries
           shape is (batch_size, input_length)
    Output: Corresponding embedding vectors of the input indexes
            shape is (batch_size, input_length, embedding_dim)
    Arguments:
      embedding_dim: the dimension of the embedding vector
      embedding_initializer: Initializer for embedding table
    """

    def __init__(self, embedding_dim, embedding_initializer="uniform"):
        super(EdlEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding_initializer = embedding_initializer
        self.tape = None
        self.worker = None
        self.bet_ids_pair = []

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape + (self.embedding_dim,)

    @property
    def name(self):
        return self._name

    def call(self, input):
        ids = tf.convert_to_tensor(input, name="embedding_ids")
        flat_ids = tf.reshape(ids, [-1])
        unique_ids, idx = tf.unique(flat_ids)
        batch_embedding = self.worker.embedding_lookup(
            unique_ids.numpy(), self._name, self.embedding_initializer
        )
        batch_embedding_tensor = tf.convert_to_tensor(batch_embedding)
        if self.tape:
            self.tape.watch(batch_embedding_tensor)
            self.bet_ids_pair.append((batch_embedding_tensor, unique_ids))
        outputs = tf.gather(batch_embedding_tensor, idx)
        outputs = tf.reshape(
            outputs, ids.get_shape().concatenate(self.embedding_dim)
        )
        return outputs

    def reset(self):
        self.bet_ids_pair = []
        self.tape = None

    def set_tape(self, tape):
        self.tape = tape

    def set_worker(self, worker):
        self.worker = worker
