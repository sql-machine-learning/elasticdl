import numpy as np
import tensorflow as tf


class EmbeddingTable(object):
    """
    EmbeddingTable is used to store embedding parameters of an embedding
    layer. The name of an embedding table is actually the embedding layer
    name. It uses a dictionary to store embedding vectors, the key is the
    item id, the value is a 1-D numpy.ndarray.

    Embedding vectors are lazily initialized in parameter server.
    EmbeddingTable also has dim and initializer fields. Inside the get
    interface of EmbeddingTable, if the id is not in the embedding_vectors
    dictionary, the corresponding value will be initialized.
    """

    def __init__(self, name, dim=None, initializer=None, is_slot=False):
        self.name = name
        self.dim = dim
        if is_slot:
            initializer = float(initializer)
            self.initializer = tf.keras.initializers.Constant(initializer)
        else:
            self.initializer = tf.keras.initializers.get(initializer)
        self.is_slot = is_slot
        self.embedding_vectors = {}

    def get(self, indices):
        if len(indices) == 0:
            return None
        values = []
        for i in indices:
            value = self.embedding_vectors.get(i, None)
            if value is None:
                value = self.initializer(shape=(self.dim,)).numpy()
                self.embedding_vectors[i] = value
            values.append(value)
        return np.stack(values)

    def set(self, indices, values):
        # TODO(qijun) need to add a RWLock in Sync-SGD
        for index, i in enumerate(indices):
            embedding_vector = values[index]
            self.embedding_vectors[i] = embedding_vector

    def clear(self):
        self.embedding_vectors.clear()


def create_embedding_table(embedding_table_info_pb):
    name = embedding_table_info_pb.name
    dim = embedding_table_info_pb.dim
    initializer = embedding_table_info_pb.initializer
    return EmbeddingTable(name, dim, initializer)
