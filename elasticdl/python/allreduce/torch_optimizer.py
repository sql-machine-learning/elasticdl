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

from horovod.torch.compression import Compression
from horovod.torch.mpi_ops import Average, allreduce_async_, size
from horovod.torch.optimizer import _DistributedOptimizer


class _ElasticDistributedOptimizer(_DistributedOptimizer):
    def __init__(
        self,
        params,
        named_parameters=None,
        compression=Compression.none,
        backward_passes_per_step=1,
        op=Average,
        gradient_predivide_factor=1.0,
        batch_num_per_step=None,
        fixed_batch_size=False,
    ):
        super(self.__class__, self).__init__(
            params,
            named_parameters,
            compression,
            backward_passes_per_step,
            op,
            gradient_predivide_factor,
        )

        self._iter_step = 0
        self._batch_num_per_step = batch_num_per_step
        self._fixed_batch_size = fixed_batch_size
        if self._fixed_batch_size and self._batch_num_per_step is None:
            raise ValueError(
                "batch_num_per_step cannot be None"
                "if fixed_batch_size is True."
            )
        self._update_gradients = True

    def _allreduce_grad_async(self, p):
        name = self._parameter_names.get(p)
        tensor = p.grad
        tensor_compressed, ctx = self._compression.compress(tensor)

        if self.op == Average:
            # Split average operation across pre/postscale factors
            # C++ backend will apply additional 1 / size() factor
            # to postscale_factor for op == Average.
            if self._fixed_batch_size:
                prescale_factor = 1.0 / (
                    self.gradient_predivide_factor * self._batch_num_per_step
                )
                postscale_factor = self.gradient_predivide_factor * size()
            else:
                prescale_factor = 1.0 / self.gradient_predivide_factor
                postscale_factor = self.gradient_predivide_factor * size()
        else:
            prescale_factor = 1.0
            postscale_factor = 1.0

        handle = allreduce_async_(
            tensor_compressed,
            name=name,
            op=self.op,
            prescale_factor=prescale_factor,
            postscale_factor=postscale_factor,
        )
        return handle, ctx

    def step(self, closure=None):
        self._iter_step += 1
        if (
            self._fixed_batch_size
            and self._iter_step % self.backward_passes_per_step != 0
        ):
            self._update_gradients = False
        else:
            self._update_gradients = True

        if self._update_gradients:
            return super().step(closure)

    def zero_grad(self):
        if self._update_gradients:
            super().zero_grad()


def ElasticDistributedOptimizer(
    optimizer,
    named_parameters=None,
    compression=Compression.none,
    backward_passes_per_step=1,
    op=Average,
    gradient_predivide_factor=1.0,
    batch_num_per_step=None,
    fixed_batch_size=False,
):

    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_ElasticDistributedOptimizer.__dict__),
    )
    return cls(
        optimizer.param_groups,
        named_parameters,
        compression,
        backward_passes_per_step,
        op,
        gradient_predivide_factor,
        batch_num_per_step,
        fixed_batch_size,
    )
