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
import os
import warnings
from contextlib import contextmanager

import torch
from horovod.torch.compression import Compression
from horovod.torch.mpi_ops import Average, allreduce_async_, size, synchronize


class _DistributedOptimizer(torch.optim.Optimizer):
    """The optimizer is implemented based on _DistributedOptimizer in Horovod
    (https://github.com/horovod/horovod/blob/v0.20.0/horovod/torch/optimizer.py).
    But it modifies the `step`, `zero_grad` and `_allreduce_grad_async` to
    be able to keep the fixed global batch size when the worker number changes.
    """

    def __init__(
        self,
        params,
        named_parameters=None,
        compression=Compression.none,
        backward_passes_per_step=1,
        op=Average,
        gradient_predivide_factor=1.0,
        global_batch_num_per_step=None,
        fixed_global_batch_size=False,
    ):
        super(self.__class__, self).__init__(params)
        self._compression = compression

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = [
                ("allreduce.noname.%s" % i, v)
                for param_group in self.param_groups
                for i, v in enumerate(param_group["params"])
            ]
        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError(
                "named_parameters should be a sequence of "
                "tuples (name, parameter), usually produced by "
                "model.named_parameters()."
            )

        dups = _DistributedOptimizer.find_duplicates(
            [k for k, _ in named_parameters]
        )
        if len(dups) > 0:
            raise ValueError(
                "Parameter names in named_parameters must be unique. "
                "Found duplicates: %s" % ", ".join(dups)
            )

        all_param_ids = {
            id(v)
            for param_group in self.param_groups
            for v in param_group["params"]
        }
        named_param_ids = {id(v) for k, v in named_parameters}
        unnamed_param_ids = all_param_ids - named_param_ids
        if len(unnamed_param_ids):
            raise ValueError(
                "named_parameters was specified, but one or more model "
                "parameters were not named. Python object ids: "
                "%s" % ", ".join(str(id) for id in unnamed_param_ids)
            )

        self._parameter_names = {v: k for k, v in sorted(named_parameters)}
        self.backward_passes_per_step = backward_passes_per_step
        self._allreduce_delay = {
            v: self.backward_passes_per_step
            for _, v in sorted(named_parameters)
        }
        self.op = op
        self.gradient_predivide_factor = gradient_predivide_factor
        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self._synchronized = False
        self._should_synchronize = True
        if size() > 1 or os.environ.get("HOROVOD_ELASTIC") == "1":
            self._register_hooks()

        self._fixed_global_batch_size = fixed_global_batch_size
        self._global_batch_num_per_step = global_batch_num_per_step
        self._iter_step = 0
        self._update_gradients = True

    def load_state_dict(self, *args, **kwargs):
        self._handles = {}
        self._synchronized = False
        self._should_synchronize = True
        for p in self._allreduce_delay:
            self._allreduce_delay[p] = self.backward_passes_per_step
        super(self.__class__, self).load_state_dict(*args, **kwargs)

    @staticmethod
    def find_duplicates(lst):
        seen = set()
        dups = set()
        for el in lst:
            if el in seen:
                dups.add(el)
            seen.add(el)
        return dups

    def set_backward_passes_per_step(self, passes):
        self.backward_passes_per_step = passes
        for p in self._allreduce_delay:
            self._allreduce_delay[p] = self.backward_passes_per_step

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group["params"]:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _allreduce_grad_async(self, p):
        name = self._parameter_names.get(p)
        tensor = p.grad
        tensor_compressed, ctx = self._compression.compress(tensor)

        if self.op == Average:
            # Split average operation across pre/postscale factors
            # C++ backend will apply additional 1 / size() factor
            # to postscale_factor for op == Average.
            if self._fixed_global_batch_size:
                # Set global_batch_num_per_step into the divisor
                # to averager gradient.
                prescale_factor = 1.0 / (
                    self.gradient_predivide_factor
                    * self._global_batch_num_per_step
                )
                # Set size() to the multiplier because C++ backend of Horovod
                # will apply additional 1 / size() factor.
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

    def _make_hook(self, p):
        def hook(*ignore):
            if p in self._handles and self._handles[p][0] is not None:
                if self._allreduce_delay[p] <= 0:
                    raise AssertionError(
                        "Gradients were computed more than "
                        "backward_passes_per_step times before call "
                        "to step(). Increase backward_passes_per_step to "
                        "accumulate gradients locally."
                    )
            assert not p.grad.requires_grad
            assert self._allreduce_delay[p] > 0
            handle, ctx = None, None
            self._allreduce_delay[p] -= 1
            if self._allreduce_delay[p] == 0:
                handle, ctx = self._allreduce_grad_async(p)
            self._handles[p] = (handle, ctx)

        return hook

    def synchronize(self):
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            handle, ctx = self._allreduce_grad_async(p)
            self._handles[p] = (handle, ctx)

        for p, (handle, ctx) in self._handles.items():
            if handle is None:
                handle, ctx = self._allreduce_grad_async(p)
                self._handles[p] = (handle, ctx)
        for p, (handle, ctx) in self._handles.items():
            output = synchronize(handle)
            self._allreduce_delay[p] = self.backward_passes_per_step
            p.grad.set_(self._compression.decompress(output, ctx))
        self._handles.clear()

        self._synchronized = True

    @contextmanager
    def skip_synchronize(self):
        """
        A context manager used to specify that optimizer.step() should
        not perform synchronization.
        It's typically used in a following pattern:
        .. code-block:: python
            optimizer.synchronize()
            with optimizer.skip_synchronize():
                optimizer.step()
        """
        self._should_synchronize = False
        try:
            yield
        finally:
            self._should_synchronize = True

    def step(self, closure=None):
        self._iter_step += 1
        if (
            self._fixed_global_batch_size
            and self._iter_step % self.backward_passes_per_step != 0
        ):
            self._update_gradients = False
        else:
            self._update_gradients = True

        if not self._update_gradients:
            return

        if self._should_synchronize:
            if self._synchronized:
                warnings.warn(
                    "optimizer.step() called without "
                    "optimizer.skip_synchronize() context after "
                    "optimizer.synchronize(). This can cause training "
                    "slowdown. You may want to consider using "
                    "optimizer.skip_synchronize() context if you use "
                    "optimizer.synchronize() in your code."
                )
            self.synchronize()
        self._synchronized = False
        return super(self.__class__, self).step(closure)

    def zero_grad(self):
        if not self._update_gradients:
            return
        if self._handles:
            raise AssertionError(
                "optimizer.zero_grad() was called after loss.backward() "
                "but before optimizer.step() or optimizer.synchronize(). "
                "This is prohibited as it can cause a race condition."
            )
        return super(self.__class__, self).zero_grad()


def DistributedOptimizer(
    optimizer,
    named_parameters=None,
    compression=Compression.none,
    backward_passes_per_step=1,
    op=Average,
    gradient_predivide_factor=1.0,
    global_batch_num_per_step=None,
    fixed_global_batch_size=False,
):
    global_batch_num_per_step = (
        global_batch_num_per_step
        if global_batch_num_per_step
        else int(os.getenv("WORKER_NUM", 1))
    )

    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedOptimizer.__dict__),
    )
    return cls(
        optimizer.param_groups,
        named_parameters,
        compression,
        backward_passes_per_step,
        op,
        gradient_predivide_factor,
        global_batch_num_per_step,
        fixed_global_batch_size,
    )
