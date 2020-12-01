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

import grpc
import numpy as np

from elasticdl.proto import elasticdl_pb2, elasticdl_pb2_grpc
from elasticdl.python.common.grpc_utils import build_channel
from elasticdl.python.common.hash_utils import (
    int_to_id,
    scatter_embedding_vector,
    string_to_id,
)
from elasticdl.python.common.tensor_utils import (
    Tensor,
    deduplicate_indexed_slices,
    merge_indexed_slices,
    pb_to_ndarray,
    serialize_indexed_slices,
    serialize_ndarray,
)

CONNECT_PS_MAX_RETRIES = 3
CONNECT_PS_TIMEOUT = 300


def build_ps_client(ps_addrs, logger):
    """
    Build a PSClient from the address list.
    Args:
        ps_addrs: a string of common separated format that stands for a list
            of address for parameter servers
        logger: a logger object
    Returns:
        A PS Client.
    """
    if not ps_addrs:
        return None

    ps_channels = []
    ps_addrs = ps_addrs.split(",")

    for addr in ps_addrs:
        # addr is in the form as "ps-pod-name.namespace.svc:port"
        channel = build_channel(addr)

        succeeded = False
        for i in range(CONNECT_PS_MAX_RETRIES):
            try:
                grpc.channel_ready_future(channel).result(
                    timeout=CONNECT_PS_TIMEOUT
                )
                logger.info(
                    "grpc channel %s to connect pod %s is ready"
                    % (addr, addr.split(".")[0])
                )
                ps_channels.append(channel)
                succeeded = True
                break
            except grpc.FutureTimeoutError:
                logger.warning(
                    "Failed to connect pod %s with %d retry"
                    % (addr.split(".")[0], i)
                )

        if not succeeded:
            raise TimeoutError(
                "Time out to connect pod %s with 3 retries"
                % addr.split(".")[0]
            )

    ps_client = PSClient(ps_channels)

    return ps_client


class PSClient(object):
    def __init__(self, ps_channels):
        self.ps_stubs = [
            elasticdl_pb2_grpc.PserverStub(c) for c in ps_channels
        ]
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
            if self.parameter_to_ps[p.name] == ps_id:
                serialize_ndarray(p.values, model.dense_parameters[p.name])
        self.ps_stubs[ps_id].push_model(model)

    def pull_dense_parameters(self, ps_ids, model_versions):
        """
        Pull dense parameters from PS.
        """
        variable_future_and_id_pairs = []
        for ps_id in ps_ids:
            if ps_id not in self.ps_to_parameter:
                continue
            stub = self.ps_stubs[ps_id]
            # async grpc call
            req = elasticdl_pb2.PullDenseParametersRequest()
            req.version = model_versions[ps_id]
            var_future = stub.pull_dense_parameters.future(req)
            variable_future_and_id_pairs.append((var_future, ps_id))

        dense_params = {}
        uninit_ps = []

        for var_future, ps_id in variable_future_and_id_pairs:
            res = var_future.result()
            if not res.initialized:
                uninit_ps.append(ps_id)
            else:
                for name, pb in res.dense_parameters.items():
                    dense_params[name] = pb_to_ndarray(pb)
                model_versions[ps_id] = res.version

        return dense_params, uninit_ps

    def push_gradients(
        self, grads, edl_grads, learning_rate, model_versions,
    ):
        """
        Push gradients to PS. There two kinds of gradients:
         - gradients of normal layers
         - sparse gradients of ElasticDL embedding layers
        """
        reqs = [
            elasticdl_pb2.PushGradientsRequest() for i in range(self.ps_num)
        ]
        ps_grads = {}

        # 1. handle grads
        for grad in grads:
            ps_id = self.parameter_to_ps[grad.name]
            if ps_id not in ps_grads:
                ps_grads[ps_id] = {grad.name: grad}
            else:
                if grad.name not in ps_grads[ps_id]:
                    ps_grads[ps_id][grad.name] = grad
                else:
                    if grad.indices is not None:
                        ps_grads[ps_id][grad.name] = merge_indexed_slices(
                            ps_grads[ps_id][grad.name], grad
                        )
                    else:
                        ps_grads[ps_id][grad.name].values += grad.values

        for ps_id, pair in ps_grads.items():
            for name, grad in pair.items():
                if grad.indices is not None:
                    v, i = deduplicate_indexed_slices(
                        grad.values, grad.indices
                    )
                    ps_grads[ps_id][name] = Tensor(None, v, i)

        for ps_id in ps_grads:
            req = reqs[ps_id]
            for name, grad in ps_grads[ps_id].items():
                # Keras embedding layer has a dense parameter,
                # but an indexed slices type gradient
                if grad.indices is not None:
                    serialize_indexed_slices(
                        Tensor(None, grad.values, grad.indices),
                        req.gradients.embedding_tables[name],
                    )
                else:
                    serialize_ndarray(
                        grad.values, req.gradients.dense_parameters[name]
                    )

        # 2. handle sparse grads of elasticdl embedding layers
        groups = {}
        for grad in edl_grads:
            if grad.name not in groups:
                groups[grad.name] = grad
            else:
                groups[grad.name] = merge_indexed_slices(
                    groups[grad.name], grad
                )

        # Sum up the values of the duplicated indices in the
        # gradients. It can reduce the gradient payload of the
        # dense embedding.
        for name, grad in groups.items():
            v, i = deduplicate_indexed_slices(grad.values, grad.indices)
            groups[name] = Tensor(None, v, i)

            results = scatter_embedding_vector(
                groups[name].values, groups[name].indices, self.ps_num
            )

            for ps_id in results:
                req = reqs[ps_id]
                gv, gi = results[ps_id]
                serialize_indexed_slices(
                    Tensor(None, gv, gi), req.gradients.embedding_tables[name],
                )

        # 3. push gradients to PS
        report_futures = []
        for ps_id in range(self.ps_num):
            req = reqs[ps_id]
            req.gradients.version = model_versions[ps_id]
            req.learning_rate = learning_rate
            report_future = self.ps_stubs[ps_id].push_gradients.future(req)
            report_futures.append(report_future)

        accepted = False
        max_version = -1
        for report_future in report_futures:
            res = report_future.result()
            if res.accepted:
                accepted = True
            if res.version > max_version:
                max_version = res.version
        return accepted, max_version

    def push_embedding_table_infos(self, infos):
        model = elasticdl_pb2.Model()
        embedding_infos = model.embedding_table_infos

        for info in infos:
            embedding_info = embedding_infos.add()
            embedding_info.name = info.name
            embedding_info.dim = info.dim
            embedding_info.initializer = info.initializer
            embedding_info.dtype = info.dtype

        for ps_id in range(self.ps_num):
            self.ps_stubs[ps_id].push_embedding_table_infos(model)
