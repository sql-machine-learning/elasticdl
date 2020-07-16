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

import numpy as np

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.hash_utils import int_to_id
from elasticdl.python.common.tensor_utils import pb_to_ndarray


class PSClient(object):
    def __init__(self, ps_stubs):
        self._ps_stubs = ps_stubs
        self._ps_num = len(self._ps_stubs)

    def pull_embedding_vectors(self, layer_name, embedding_ids):
        """
        Pulls and returns embedding vectors ordered by the embedding ids.
        Args:
            layer_name: layer name
            embedding_ids: a list of ids
        Return:
            embedding_vectors: a 2-D numpy ndarray
        """
        ps_ids = {}
        ps_ids_index = {}
        for idx, embedding_id in enumerate(embedding_ids):
            ps_id = int_to_id(embedding_id, self._ps_num)
            ps_ids.setdefault(ps_id, []).append(embedding_id)
            ps_ids_index.setdefault(ps_id, []).append(idx)

        embeddings = []
        index = []
        pb_future_and_id_pairs = []
        for ps_id, embedding_ids in ps_ids.items():
            req = elasticdl_pb2.PullEmbeddingVectorRequest()
            req.name = layer_name
            req.ids.extend(embedding_ids)
            pb_future = self._ps_stubs[ps_id].pull_embedding_vectors.future(
                req
            )
            pb_future_and_id_pairs.append((pb_future, ps_id))
        for pb_future, ps_id in pb_future_and_id_pairs:
            pb = pb_future.result()
            embeddings.append(pb_to_ndarray(pb))
            index.extend(ps_ids_index[ps_id])
        embeddings = np.concatenate(embeddings)

        # adjust the order of embedding vectors
        new_embeddings = np.empty_like(embeddings)
        new_embeddings[index] = embeddings
        return new_embeddings
