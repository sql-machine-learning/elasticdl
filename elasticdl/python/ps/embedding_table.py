# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading

import numpy as np
import tensorflow as tf

from elasticdl.proto.elasticdl_pb2 import EmbeddingTableInfo
from elasticdl.python.common.dtypes import dtype_numpy_to_tensor


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
        """
        Args:
            name: The embedding table name.
            dim: The dimension of embeddings in this embedding table.
            initializer: The initializer to initialize new embeddings. If this
                embedding table is for slots, `initializer` is a float and this
                table will initialize with constant initializer. Otherwise
                `initializer` is the name of Keras initializer.
            is_slot: A bool. True for storing slot variable, otherwise false.
        """
        self.name = name
        self.dim = dim
        self.initializer_value = initializer
        # set dtype to float32
        self.dtype = np.dtype("float32")
        if is_slot:
            self.initializer = tf.keras.initializers.Constant(
                float(self.initializer_value)
            )
        else:
            self.initializer = tf.keras.initializers.get(
                self.initializer_value
            )
        self.is_slot = is_slot
        self.embedding_vectors = {}
        self._lock = threading.Lock()

    def get(self, indices):
        if len(indices) == 0:
            return None
        values = []
        for i in indices:
            value = self.embedding_vectors.get(i, None)
            if value is None:
                with self._lock:
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

    def to_indexed_slices(self):
        indices = []
        embedding_vectors = []
        with self._lock:
            for id, embedding_vector in self.embedding_vectors.items():
                indices.append(id)
                embedding_vectors.append(embedding_vector)
        return tf.IndexedSlices(
            values=np.array(embedding_vectors), indices=np.array(indices)
        )

    def to_embedding_table_info_pb(self):
        """Convert the embedding table information to a protobuf"""
        embedding_pb = EmbeddingTableInfo()
        embedding_pb.name = self.name
        embedding_pb.dim = self.dim
        embedding_pb.initializer = str(self.initializer_value)
        embedding_pb.dtype = dtype_numpy_to_tensor(self.dtype)
        return embedding_pb

    def get_table_size(self):
        """Get the element count of an embedding table"""
        if len(self.embedding_vectors) > 0:
            element_size = list(self.embedding_vectors.values())[0].itemsize
            size = self.dim * len(self.embedding_vectors) * element_size
            return size
        return 0

    def debug_info(self):
        return (
            "Embedding param name: %s\n  shape: [%d, %d]\n  size: %d bytes\n"
            % (
                self.name,
                len(self.embedding_vectors),
                self.dim,
                self.get_table_size(),
            )
        )


# TODO(bug): create_embedding_table does not create EmbeddingTable correctly
#     if it is a slot table.
def create_embedding_table(embedding_table_info_pb):
    name = embedding_table_info_pb.name
    dim = embedding_table_info_pb.dim
    initializer = embedding_table_info_pb.initializer
    return EmbeddingTable(name, dim, initializer)


def get_slot_table_name(embedding_name, slot_name):
    return embedding_name + "-" + slot_name
