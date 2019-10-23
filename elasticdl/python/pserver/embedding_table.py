import numpy as np
import tensorflow as tf


class EmbeddingTable(object):
    def __init__(self, name, dim=None, initializer=None):
        self.name = name
        self.dim = dim
        self.initializer = initializer
        self.embedding_vectors = {}

    def get(self, indices):
        if len(indices) == 0:
            return None
        values = []
        for i in indices:
            if i not in self.embedding_vectors:
                init = tf.keras.initializers.get(self.initializer)
                value = init(shape=self.dim).numpy()
                self.embedding_vectors[i] = value
            else:
                value = self.embedding_vectors[i]
            values.append(value)
        return np.stack(values)

    def set(self, indices, values):
        for index, i in enumerate(indices):
            embedding_vector = values[index, :]
            self.embedding_vectors[i] = embedding_vector

    def clear(self):
        self.embedding_vectors.clear()


def create_embedding_table(embedding_table_info_pb):
    name = embedding_table_info_pb.name
    dim = tuple(embedding_table_info_pb.dim)
    initializer = embedding_table_info_pb.initializer
    return EmbeddingTable(name, dim, initializer)
