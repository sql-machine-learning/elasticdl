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
from elasticdl.python.common.hash_utils import int_to_id, string_to_id
from elasticdl.python.common.tensor_utils import (
    Tensor,
    pb_to_ndarray,
    serialize_ndarray,
)


class PSClient(object):
    def __init__(self, ps_stubs):
        self.ps_stubs = ps_stubs
        self.ps_num = len(self.ps_stubs)
        self.parameter_to_ps = {}
        self.ps_to_parameter = {}

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
            ps_id = int_to_id(embedding_id, self.ps_num)
            ps_ids.setdefault(ps_id, []).append(embedding_id)
            ps_ids_index.setdefault(ps_id, []).append(idx)

        embeddings = []
        index = []
        pb_future_and_id_pairs = []
        for ps_id, embedding_ids in ps_ids.items():
            req = elasticdl_pb2.PullEmbeddingVectorRequest()
            req.name = layer_name
            req.ids.extend(embedding_ids)
            pb_future = self.ps_stubs[ps_id].pull_embedding_vectors.future(req)
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

    def partition_dense_parameters(self, param_names):
        """
        Partition dense parameters to PS
        ps_id = string_to_id(param_name)
        """
        for name in param_names:
            if name not in self.parameter_to_ps:
                self.parameter_to_ps[name] = string_to_id(name, self.ps_num)
                ps_id = self.parameter_to_ps[name]
                if ps_id not in self.ps_to_parameter:
                    self.ps_to_parameter[ps_id] = [name]
                else:
                    self.ps_to_parameter[ps_id].append(name)

    def push_dense_parameters(self, parameters, ps_id, version):
        """
        Push dense parameters to PS
        Args:
            parameters: a list of Tensors
            ps_id: PS id
            version: model version
        """

        model = elasticdl_pb2.Model()
        model.version = version
        for p in parameters:
            serialize_ndarray(p.values, model.dense_parameters[p.name])
        self.ps_stubs[ps_id].push_model(model)

    def pull_dense_parameters(self, model_versions, dense_params):
        variable_future_and_id_pairs = []
        for ps_id, stub in enumerate(self.ps_stubs):
            if ps_id not in self.ps_to_parameter:
                continue
            # async grpc call
            req = elasticdl_pb2.PullDenseParametersRequest()
            req.version = model_versions[ps_id]
            var_future = stub.pull_dense_parameters.future(req)
            variable_future_and_id_pairs.append((var_future, ps_id))

        for var_future, ps_id in variable_future_and_id_pairs:
            res = var_future.result()
            if not res.initialized:
                # push variable to ps for initialization
                parameters = [
                    Tensor(name, dense_params[name].values, None)
                    for name in self.ps_to_parameter[ps_id]
                ]
                self.push_dense_parameters(
                    parameters, ps_id, model_versions[ps_id]
                )
                req = elasticdl_pb2.PullDenseParametersRequest()
                req.version = model_versions[ps_id]
                res = self.ps_stubs[ps_id].pull_dense_parameters(req)
                if not res.initialized:
                    # TODO: support PS fault-tolerance
                    raise RuntimeError(
                        "PS pod %d cannot be initialized" % ps_id
                    )

            for name, pb in res.dense_parameters.items():
                self.dense_params.values = pb_to_ndarray(pb)
            model_versions[ps_id] = res.version
