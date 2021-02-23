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

import horovod.tensorflow as hvd
import tensorflow as tf
from horovod.tensorflow import _LegacyOptimizer

optimizer_instances: list = []


def complement_value_from_env_if_none(
    original_value, key, clz, default_value=None
):
    if original_value is not None:
        return original_value

    return clz(os.environ.get(key, default_value))


def get_adjust_backward_passes_hooks():
    hooks = []
    for opt in optimizer_instances:
        hooks.append(AdjustBackwardPassesPerStepHook(opt))
    return hooks


class AdjustBackwardPassesPerStepHook(tf.train.SessionRunHook):
    """
    Hooks that adjusts `backward_passer_per_step` according to
    the horovod size dynamically.
    """

    def __init__(self, optimizer, hvd_max_size=None):
        """
        Args:
            optimizer: A DistributedOptimizer from ElasticDL.
            hvd_max_size: The maximum horovod size for the elastic training.
        """
        self._value_placeholder = tf.placeholder(
            optimizer.mutable_local_backward_passes_per_step.dtype, []
        )
        self._update_op = tf.assign(
            optimizer.mutable_local_backward_passes_per_step,
            self._value_placeholder,
        )

        hvd_max_size = complement_value_from_env_if_none(
            hvd_max_size, "WORKER_NUM", int, 1
        )
        self._global_batch_count_per_step = (
            hvd_max_size * optimizer.backward_passes_per_step
        )

    def before_run(self, run_context):
        size = hvd.size()
        rank = hvd.rank()
        backward_passes_per_step_value = (
            self._global_batch_count_per_step // size
            + int(rank < self._global_batch_count_per_step % size)
        )
        run_context.session.run(
            self._update_op,
            feed_dict={
                self._value_placeholder: backward_passes_per_step_value
            },
        )


def apply_op_to_not_none_tensors(tensor_op, tensors, *args):
    return [
        tensor_op(tensor, *args) if tensor is not None else tensor
        for tensor in tensors
    ]


def get_not_none_from_list(tensor_list):
    return [x for x in tensor_list if x is not None]


# Update based on horovod v0.21.0
# https://github.com/horovod/horovod/blob/7d71874258fc8625ad8952defad0ea5b24531248/horovod/tensorflow/gradient_aggregation.py#L16
# Make `backward_passes_per_step` a tensorflow variable instead of
# a primitive python type.
class LocalGradientAggregationHelper:
    """
    LocalGradientAggregationHelper aggregates gradient updates locally,
    and communicates the updates across machines only once every
    backward_passes_per_step. Only supports graph mode execution.
    """

    _OPTIMIZER_TYPE_KERAS = "optimizer_type_keras"
    _OPTIMIZER_TYPE_LEGACY = "optimizer_type_legacy"

    def __init__(
        self,
        backward_passes_per_step,
        allreduce_func,
        sparse_as_dense,
        average_aggregated_gradients,
        rank,
        optimizer_type,
        global_batch_count_per_step,
        op,
    ):
        self._allreduce_grads = allreduce_func

        # backward_passes_per_step is the value set in the model definition,
        self.backward_passes_per_step = backward_passes_per_step
        # mutable_local_backward_passes_per_step controls how often gradient
        # updates are synchronized for this process.
        self.mutable_local_backward_passes_per_step = tf.Variable(
            initial_value=backward_passes_per_step,
            trainable=False,
            dtype=tf.int32,
        )

        # average_aggregated_gradients controls whether gradient updates that
        # are aggregated, should be divided by `backward_passes_per_step`.
        self.average_aggregated_gradients = average_aggregated_gradients

        # This is going to be [N] data structure holding the aggregated
        # gradient updates N is the number of parameters.
        self.locally_aggregated_grads = []

        # Used to know when to allreduce and apply gradients. We allreduce
        # when `self.counter` is equal to
        # `self.mutable_local_backward_passes_per_step`. We apply gradients
        # when `self.counter` is equal to 0.
        self.counter = None

        self.sparse_as_dense = sparse_as_dense
        self.rank = rank
        self.optimizer_type = optimizer_type

        # Contains the mapping of indexes of grad updates that are not None to
        # their index in locally_aggregated_grads which only contains not None
        # gradients. When performing gradient aggregation we have to remove
        # them from the list of grads prior to passing the list into a
        # tf.cond().
        self.not_none_indexes = {}
        self.num_none_grad_updates = 0

        # *ElasticDL Update*: add two more fields for updating
        # gradient value.
        self.global_batch_count_per_step = global_batch_count_per_step
        self.op = op

    def _init_aggregation_vars(self, grads):
        """
        Initializes the counter that is used when to communicate and aggregate
        gradients and the tensorflow variables that store the locally
        aggregated gradients.
        """
        variable_scope_name = "aggregation_variables_" + str(self.rank)
        with tf.compat.v1.variable_scope(
            variable_scope_name, reuse=tf.compat.v1.AUTO_REUSE
        ):
            self.counter = tf.compat.v1.get_variable(
                "aggregation_counter",
                shape=(),
                dtype=tf.int32,
                trainable=False,
                initializer=tf.compat.v1.zeros_initializer(),
                collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES],
            )
            for idx, grad in enumerate(grads):
                # Handle IndexedSlices.
                if self.sparse_as_dense and isinstance(grad, tf.IndexedSlices):
                    grad = tf.convert_to_tensor(grad)
                elif isinstance(grad, tf.IndexedSlices):
                    raise ValueError(
                        "IndexedSlices are not supported when "
                        "`backward_passes_per_step` > 1 and "
                        "`sparse_as_dense` is False."
                    )

                # Handle grads that are None.
                if grad is None:
                    self.num_none_grad_updates += 1
                    continue
                self.not_none_indexes[idx] = len(self.locally_aggregated_grads)

                # Create shadow variable.
                grad_aggregation_variable_name = str(idx)
                zero_grad = tf.zeros(
                    shape=grad.get_shape().as_list(), dtype=grad.dtype
                )
                grad_aggregation_variable = tf.compat.v1.get_variable(
                    grad_aggregation_variable_name,
                    trainable=False,
                    initializer=zero_grad,
                    collections=[
                        tf.compat.v1.GraphKeys.LOCAL_VARIABLES,
                        "aggregating_collection",
                    ],
                )
                self.locally_aggregated_grads.append(grad_aggregation_variable)
            assert len(
                self.locally_aggregated_grads
            ) + self.num_none_grad_updates == len(grads)

        # We expect to get a `sess` when we need to manually do a
        # `sess.run(...)` for the variables to be initialized. This is the
        # `tf.keras` optimizers.
        if self.optimizer_type == self._OPTIMIZER_TYPE_KERAS:
            session = tf.compat.v1.keras.backend.get_session(op_input_list=())
            vars_init_op = tf.compat.v1.variables_initializer(
                [
                    self.counter,
                    *get_not_none_from_list(self.locally_aggregated_grads),
                ]
            )
            session.run(vars_init_op)

    def _clear_grads(self):
        clear_ops_list = []
        for idx, grad_aggregator in enumerate(self.locally_aggregated_grads):
            clear_op = grad_aggregator.assign(grad_aggregator.initial_value)
            clear_ops_list.append(clear_op)
        return tf.group(*clear_ops_list)

    def _aggregate_grads(self, grads):
        aggregation_ops_list = []
        grads = get_not_none_from_list(grads)
        assert len(grads) == len(self.locally_aggregated_grads)

        # Apply new gradient updates to the local copy.
        for idx, grad in enumerate(grads):
            if self.sparse_as_dense and isinstance(grad, tf.IndexedSlices):
                grad = tf.convert_to_tensor(grad)

            updated_grad_aggregator = self.locally_aggregated_grads[
                idx
            ].assign_add(grad)
            aggregation_ops_list.append(updated_grad_aggregator)

        return aggregation_ops_list

    def _allreduce_grads_helper(self, grads):
        # Read in latest variables values.
        aggregated_grads = []
        aggregation_read_ops_list = []
        for idx, locally_aggregated_grad in enumerate(
            self.locally_aggregated_grads
        ):
            aggregated_grads.append(locally_aggregated_grad.read_value())
            aggregation_read_ops_list.append(aggregated_grads[idx])
        aggregation_read_ops = tf.group(*aggregation_read_ops_list)

        with tf.control_dependencies([aggregation_read_ops]):
            averaged_gradients = self._allreduce_grads(aggregated_grads)

            # Reset counter.
            with tf.control_dependencies(
                [g.op for g in averaged_gradients if g is not None]
            ):
                reset_op = self.counter.assign(
                    tf.constant(0), use_locking=True
                )

            with tf.control_dependencies([reset_op]):
                # *ElasticDL Update*: If ReduceOP is average,
                # multiply horovod_size / global_batch_count_per_step.
                averaged_gradients = self.update_gradients_for_elastic_workers(
                    averaged_gradients
                )

                return averaged_gradients

    # *ElasticDL Update*: Update the gradient using
    # global_batch_count_per_step if the ReduceOP is average.
    def update_gradients_for_elastic_workers(self, grads):
        gradient_multiplier = 1
        if self.op == hvd.Average:
            valid_grads = get_not_none_from_list(grads)
            horovod_size = tf.cast(
                hvd.size_op()
                if int(os.environ.get("HOROVOD_ELASTIC", 0))
                else hvd.size(),
                dtype=valid_grads[0].dtype,
            )
            gradient_multiplier = (
                (horovod_size / self.global_batch_count_per_step)
                if self.average_aggregated_gradients
                else (
                    horovod_size
                    * self.backward_passes_per_step
                    / self.global_batch_count_per_step
                )
            )
        elif self.op == hvd.Sum:
            gradient_multiplier = (
                1.0 / self.backward_passes_per_step
                if self.average_aggregated_gradients
                else 1
            )

        return apply_op_to_not_none_tensors(
            tf.multiply, grads, gradient_multiplier
        )

    def compute_gradients(self, grads):
        """
        Applies the new gradient updates the locally aggregated gradients, and
        performs cross-machine communication every backward_passes_per_step
        times it is called.
        """
        self._init_aggregation_vars(grads)

        # Clear the locally aggregated gradients when the counter is at zero.
        clear_op = tf.cond(
            pred=tf.equal(self.counter, 0),
            true_fn=lambda: self._clear_grads(),
            false_fn=tf.no_op,
        )

        # Add new gradients to the locally aggregated gradients.
        with tf.control_dependencies([clear_op]):
            aggregation_ops_list = self._aggregate_grads(grads)

        # Increment the counter once new gradients have been applied.
        aggregation_ops = tf.group(*aggregation_ops_list)
        with tf.control_dependencies([aggregation_ops]):
            update_counter = self.counter.assign_add(tf.constant(1))

        with tf.control_dependencies([update_counter]):
            grads = get_not_none_from_list(grads)
            assert len(grads) == len(self.locally_aggregated_grads)

            # Allreduce locally aggregated gradients when the counter equals
            # or exceeds backward_passes_per_step. The counter may exceed
            # backward_passes_per_step because of retries in the fault-tolerant
            # allreduce. When the condition is true, it also resets the counter
            # back to 0.
            allreduced_grads = tf.cond(
                tf.math.less(
                    self.counter, self.mutable_local_backward_passes_per_step
                ),
                lambda: grads,
                lambda: self._allreduce_grads_helper(grads),
            )

            # Handle case where there is only one variable.
            if not isinstance(allreduced_grads, (list, tuple)):
                allreduced_grads = (allreduced_grads,)
            assert len(allreduced_grads) == len(self.locally_aggregated_grads)

            # Insert gradients that are None back in.
            allreduced_grads = [
                allreduced_grads[self.not_none_indexes[idx]]
                if idx in self.not_none_indexes
                else None
                for idx in range(
                    len(self.locally_aggregated_grads)
                    + self.num_none_grad_updates
                )
            ]
            assert (
                len(allreduced_grads)
                == len(self.locally_aggregated_grads)
                + self.num_none_grad_updates
            )

        # If gradients have not been allreduced this batch, we return the
        # gradients that were submitted as the updates (the input).
        return allreduced_grads

    def apply_gradients(self, apply_grads_closure, optimizer, *args, **kwargs):
        """
        Apply updates every backward_passes_per_step, which lines up with
        the batches on which we communicated the locally aggregated gradients.
        """
        flattended_args0 = [item for tup in args[0] for item in tup]

        # Since we skip applying updates when the counter is not at zero we
        # still want to increment the global step if it is being tracked
        # (e.g., Tensorflow Estimators).
        def increment_global_step_counter():
            global_step_counter = tf.compat.v1.train.get_global_step()
            if global_step_counter is None:
                return tf.no_op()
            return global_step_counter.assign_add(
                tf.constant(1, dtype=tf.int64),
                use_locking=True,
                read_value=False,
            )

        # Increment global step on iterations where we don't call
        # `apply_gradients()`.
        cond_increment_global_step_counter = tf.cond(
            pred=tf.equal(self.counter, 0),
            true_fn=tf.no_op,
            false_fn=increment_global_step_counter,
        )
        flattended_args0.append(cond_increment_global_step_counter)

        # If optimizer tracks iterations, we increment it on steps where we
        # are not going to call `apply_gradients()`.
        def increment_optimizer_iteration():
            if (
                hasattr(optimizer, "_iterations")
                and optimizer._iterations is not None
            ):
                return optimizer._iterations.assign_add(1).op
            return tf.no_op()

        with tf.control_dependencies(
            [tf.group(*get_not_none_from_list(flattended_args0))]
        ):
            return tf.cond(
                pred=tf.equal(self.counter, 0),
                true_fn=apply_grads_closure,
                false_fn=increment_optimizer_iteration,
            )


# `_DistributedOptimizer` is updated based on horovod v0.21.0
# https://github.com/horovod/horovod/blob/7d71874258fc8625ad8952defad0ea5b24531248/horovod/tensorflow/__init__.py#L396
# Update the __init__ function.
class _DistributedOptimizer(_LegacyOptimizer):
    """An optimizer that wraps another tf.Optimizer, using an allreduce to
    combine gradient values before applying gradients to model weights."""

    def __init__(
        self,
        optimizer,
        name=None,
        use_locking=False,
        device_dense="",
        device_sparse="",
        compression=hvd.Compression.none,
        sparse_as_dense=False,
        op=hvd.Average,
        gradient_predivide_factor=1.0,
        backward_passes_per_step=1,
        average_aggregated_gradients=False,
        num_groups=0,
        global_batch_count_per_step=None,
    ):
        if name is None:
            name = "Distributed{}".format(type(optimizer).__name__)
        super(_DistributedOptimizer, self).__init__(
            name=name, use_locking=use_locking
        )

        self._optimizer = optimizer
        self._allreduce_grads = hvd._make_allreduce_grads_fn(
            name,
            device_dense,
            device_sparse,
            compression,
            sparse_as_dense,
            op,
            gradient_predivide_factor,
            num_groups,
        )

        # *ElasticDL Update*: Always create LocalGradientAggregationHelper
        # for Elastic training with fixed global batch size.
        self._agg_helper = LocalGradientAggregationHelper(
            backward_passes_per_step=backward_passes_per_step,
            allreduce_func=self._allreduce_grads,
            sparse_as_dense=sparse_as_dense,
            average_aggregated_gradients=average_aggregated_gradients,
            rank=hvd.rank(),
            optimizer_type=LocalGradientAggregationHelper._OPTIMIZER_TYPE_LEGACY,  # noqa: ignore=E501
            global_batch_count_per_step=global_batch_count_per_step,
            op=op,
        )

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients of all trainable variables.

        See Optimizer.compute_gradients() for more info.

        In DistributedOptimizer, compute_gradients() is overriden to also
        allreduce the gradients before returning them.
        """
        gradients = self._optimizer.compute_gradients(*args, **kwargs)
        grads, vars = zip(*gradients)
        if self._agg_helper:
            avg_grads = self._agg_helper.compute_gradients(grads)
        else:
            avg_grads = self._allreduce_grads(grads)
        return list(zip(avg_grads, vars))

    def apply_gradients(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        if self._agg_helper:
            return self._agg_helper.apply_gradients(
                lambda: self._optimizer.apply_gradients(*args, **kwargs),
                self._optimizer,
                *args,
                **kwargs,
            )

        return self._optimizer.apply_gradients(*args, **kwargs)

    def get_slot(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.variables(*args, **kwargs)

    @property
    def backward_passes_per_step(self):
        return self._agg_helper.backward_passes_per_step

    @property
    def mutable_local_backward_passes_per_step(self):
        return self._agg_helper.mutable_local_backward_passes_per_step


def DistributedOptimizer(
    optimizer,
    name=None,
    use_locking=False,
    device_dense="",
    device_sparse="",
    compression=hvd.Compression.none,
    sparse_as_dense=False,
    backward_passes_per_step=1,
    op=hvd.Average,
    gradient_predivide_factor=1.0,
    average_aggregated_gradients=False,
    num_groups=0,
    fixed_global_batch_size=False,
    hvd_max_size=None,
):
    """Construct a new DistributedOptimizer, which uses another optimizer
    under the hood for computing single-process gradient values and
    applying gradient updates after the gradient values have been combined
    across all the Horovod ranks.

    Args:
      optimizer:
        Optimizer to use for computing gradients and applying updates.
      name:
        Optional name prefix for the operations created when applying
        gradients. Defaults to "Distributed" followed by the provided
        optimizer type.
      use_locking:
        Whether to use locking when updating variables.
        See Optimizer.__init__ for more info.
      device_dense:
        Device to be used for dense tensors. Uses GPU by default
        if Horovod was built with HOROVOD_GPU_OPERATIONS.
      device_sparse:
        Device to be used for sparse tensors. Uses GPU by default
        if Horovod was built with HOROVOD_GPU_OPERATIONS.
      compression:
        Compression algorithm used during allreduce to reduce the amount
        of data sent during each parameter update step.  Defaults to
        not using compression.
      sparse_as_dense:
        Treat all sparse gradients as dense tensors.  This can help improve
        performance and memory utilization if the original sparse gradient
        has high density.  Defaults to false.
      backward_passes_per_step:
        Number of backward passes to perform before calling hvd.allreduce.
        This allows accumulating updates over multiple mini-batches before
        reducing and applying them.
      op:
        The reduction operation to use when combining gradients across
        different ranks.
      gradient_predivide_factor:
        If op == Average, gradient_predivide_factor splits the averaging
        before and after the sum. Gradients are scaled by
        1.0 / gradient_predivide_factor before the sum and
        gradient_predivide_factor / size after the sum.
      average_aggregated_gradients:
        Whether to average the aggregated gradients that have been accumulated
        over multiple mini-batches. If true divides gradients updates by
        backward_passes_per_step. Only applicable for
        backward_passes_per_step > 1.
      num_groups:
        Number of groups to assign gradient allreduce ops to for explicit
        grouping. Defaults to no explicit groups.
      fixed_global_batch_size:
        Whether to keep the global batch size is fixed even though the worker
        number is changing during elastic execution.
      hvd_max_size:
        The maximum horovod size for the elastic training.
    """

    # *ElasticDL Update*: If `fixed_global_batch_size` == False,
    # just fallback to the native horovod DistributedOptimizer.
    if not fixed_global_batch_size:
        return hvd.DistributedOptimizer(
            optimizer=optimizer,
            name=name,
            use_locking=use_locking,
            device_dense=device_dense,
            device_sparse=device_sparse,
            compression=compression,
            sparse_as_dense=sparse_as_dense,
            backward_passes_per_step=backward_passes_per_step,
            op=op,
            gradient_predivide_factor=gradient_predivide_factor,
            average_aggregated_gradients=average_aggregated_gradients,
            num_groups=num_groups,
        )

    if gradient_predivide_factor != 1.0:
        if hvd.rocm_built():
            raise ValueError(
                "gradient_predivide_factor not supported yet with ROCm"
            )
        if op != hvd.Average:
            raise ValueError(
                "gradient_predivide_factor not supported with op != Average"
            )

    if op == hvd.Adasum and average_aggregated_gradients:
        raise ValueError(
            "Adasum does not support average_aggregated_gradients == True"
        )

    if isinstance(optimizer, _LegacyOptimizer):
        if op == hvd.Adasum:
            raise ValueError(
                """op == Adasum and fixed_global_batch_size == True is
                not yet supported"""
            )

        hvd_max_size = complement_value_from_env_if_none(
            hvd_max_size, "WORKER_NUM", int, 1
        )
        global_batch_count_per_step = hvd_max_size * backward_passes_per_step
        opt = _DistributedOptimizer(
            optimizer=optimizer,
            name=name,
            use_locking=use_locking,
            device_dense=device_dense,
            device_sparse=device_sparse,
            compression=compression,
            sparse_as_dense=sparse_as_dense,
            op=op,
            gradient_predivide_factor=gradient_predivide_factor,
            backward_passes_per_step=backward_passes_per_step,
            average_aggregated_gradients=average_aggregated_gradients,
            num_groups=num_groups,
            global_batch_count_per_step=global_batch_count_per_step,
        )
        optimizer_instances.append(opt)
        return opt
    elif isinstance(optimizer, tf.keras.optimizers.Optimizer):
        raise ValueError(
            "fixed_global_batch_size == True is not supported yet with Keras"
        )
    else:
        raise ValueError(
            "Provided optimizer doesn't inherit from either legacy "
            "TensorFlow or Keras optimizer: %s" % optimizer
        )
