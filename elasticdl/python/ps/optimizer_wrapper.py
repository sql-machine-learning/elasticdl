"""Optimizer Wrapper for ElasticDL"""

import threading

import tensorflow as tf
from tensorflow.keras.optimizers import (
    SGD,
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    Ftrl,
    Nadam,
    RMSprop,
)

from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.ps.embedding_table import get_slot_table_name


def _get_embedding_layer_name_from_var(var):
    """Get name for ElasticDL embedding layer from variable."""
    # Assumes that for ElasticDL embedding layer, variable will be a
    # string representing its layer name
    if isinstance(var, str):
        return var
    return None


# This function is taken from `tensorflow.keras.optimizers.Optimizer._var_key`.
# https://github.com/tensorflow/tensorflow/blob/71d73e56a2e66e4a6805d967cfa48ea
# 594f8c54e/tensorflow/python/keras/optimizer_v2/optimizer_v2.py#L1033
def _var_key(var):
    """Key for representing a primary variable, for looking up slots.

    In graph mode the name is derived from the var shared name.
    In eager mode the name is derived from the var unique id.
    If distribution strategy exists, get the primary variable first.

    Arguments:
        var: the variable.

    Returns:
        the unique name of the variable.
    """

    # pylint: disable=protected-access
    # Get the distributed variable if it exists.
    if getattr(var, "_distributed_container", None) is not None:
        var = var._distributed_container()
    if var._in_graph_mode:
        return var._shared_name
    return var._unique_id


class OptimizerWrapper(object):
    """ ElasticDL optimizer wrapper.

    If model does not use ElasticDL embedding layer, `OptimizerWrapper`
    does nothing but calls `apply_gradients` function of TensorFlow optimizer.
    Otherwise, `OptimizerWrapper` looks up embedding vectors and slot values
    from external kv store before updating variables, and updates embedding
    vectors and slot values in kv store after updating variables.
    """

    def __init__(
        self,
        opt,
        use_async=False,
        lookup_embedding_func=None,
        update_embedding_func=None,
    ):
        """
        Note:
            We need to support Redis and ElasticDL parameter server at the
            same time. If `lookup_embedding_func`/`update_embedding_func`
            is not None, use parameter server to lookup/update embedding.
            Otherwise use Redis.

        Arguments:
            opt: A TensorFlow optimizer instance.
            kv_store_endpoint: The endpoint to kv store.
            embedding_dims: A python dictionary of
                {layer name: `embedding_dim`} where layer name is the
                name of ElasticDL embedding layer and `embedding_dim`
                is the output dimension of corresponding embedding layer.
            use_async: A python bool. True if using asynchronous updates. When
                using asynchronoues updates, `OptimizerWrapper` is thread-safe
                for non-embedding variables and is not thread-safe for
                embedding table.
            lookup_embedding_func: The function to lookup embeddings. The
                argument of this function is a list of keys.
            update_embedding_func: The function to update embeddings. The
                arguments of this function is a key list and a value list.
        """
        self._opt = opt
        self._use_async = use_async
        self._lookup_embedding_func = lookup_embedding_func
        self._update_embedding_func = update_embedding_func
        self._slot_initial_value = {}

        self._update_gradient_lock = threading.Lock()
        self._tls = threading.local()
        self._init_thread_local()
        self._has_embedding = False

        # "-" in slot name is not supported
        if isinstance(opt, SGD):
            self._allowed_slot_names = []
            if opt._momentum:
                self._allowed_slot_names.append("momentum")

        elif isinstance(opt, (Adam, Adamax, Nadam)):
            self._allowed_slot_names = ["m", "v"]
            if isinstance(opt, Adam) and self._opt.amsgrad:
                self._allowed_slot_names.append("vhat")

        elif isinstance(opt, Adadelta):
            self._allowed_slot_names = ["accum_grad", "accum_var"]

        elif isinstance(opt, (Adagrad, Ftrl)):
            self._allowed_slot_names = ["accumulator"]
            if isinstance(opt, Ftrl):
                self._allowed_slot_names.append("linear")
            accumu_init = opt._initial_accumulator_value
            self._slot_initial_value["accumulator"] = accumu_init

        elif isinstance(opt, RMSprop):
            self._allowed_slot_names = ["rms"]
            if self._opt._momentum:
                self._allowed_slot_names.append("momentum")
            if self._opt.centered:
                self._allowed_slot_names.append("mg")

        else:
            raise NotImplementedError(
                "Optimizer %s is not supported in ElasticDL." % type(opt)
            )

        for slot in self._allowed_slot_names:
            self._slot_initial_value.setdefault(slot, 0.0)

    def _init_thread_local(self):
        self._tls._unique_ids_all_layers = {}
        self._tls._embed_variables = {}
        self._tls._slot_variables = {}

    def apply_gradients(self, grads_and_vars):
        """Update variable values.

        Args:
            grads_and_vars: A list of (gradient, variable) pairs. If the
                variable is from ElasticDL embedding layer, it should be a
                ElasticDL `Tensor` object. Otherwise it is a TensorFlow
                variable.
        """
        self._init_thread_local()

        if self._has_embedding:
            with self._update_gradient_lock:
                self._update_parameter_by_gradients(grads_and_vars)
        else:
            self._update_parameter_by_gradients(grads_and_vars)

    def _update_parameters_by_gradients(self, grads_and_vars):
        """Update parameters by gradients received by GRPC"""
        grads_and_vars_new = []
        for grad, var in grads_and_vars:
            # If var is a string, create the grad var pair for
            # ElasticDL embedding
            if isinstance(var, str):
                grads_and_vars_new.append(
                    self._get_embedding_var_and_grad(grad, var)
                )
                self._has_embedding = True
            else:
                grads_and_vars_new.append((grad, var))
        self._opt.apply_gradients(grads_and_vars_new)
        self._update_embedding_param()
        self._delete_variables()

    def _get_embedding_var_and_grad(self, grad, layer_name):
        unique_ids, indices = tf.unique(grad.indices)
        unique_ids = unique_ids.numpy()
        if layer_name in self._tls._unique_ids_all_layers:
            # TODO: support grads_and_vars with duplicated layer name
            logger.warning(
                "grads_and_vars has duplicated layer name %s." % layer_name
            )
        self._tls._unique_ids_all_layers[layer_name] = unique_ids
        new_grad = tf.IndexedSlices(values=grad.values, indices=indices)

        embed_value = self._lookup_embedding_func(layer_name, unique_ids)
        embed_var = self._create_embedding_variable(layer_name, embed_value)
        self._get_slot_and_set_to_optimizer(layer_name)
        return new_grad, embed_var

    def _create_embedding_variable(self, name, initial_value):
        """Creates a TensorFlow variable using given initial value.

        Note that this function saves the created variable to
        `self._tls._embed_variables`.
        """
        embed_var = tf.Variable(
            initial_value,
            name=name + str(threading.get_ident()),
            shape=initial_value.shape,
            dtype=tf.float32,
            trainable=False,
        )
        self._tls._embed_variables[name] = embed_var
        return embed_var

    def _get_slot_and_set_to_optimizer(self, layer_name):
        """Looks up slot value and set it to TensorFlow optimizer."""
        for slot_name in self._allowed_slot_names:
            param_name = get_slot_table_name(layer_name, slot_name)
            indices = self._tls._unique_ids_all_layers[layer_name]
            slot_value = self._lookup_embedding_func(param_name, indices)
            # self._create_slot_variable creates a slot variable in tf
            # optimizer and set slot_value to it.
            self._create_slot_variable(layer_name, slot_name, slot_value)

    def _get_slot_variable(self, layer_name, slot_name):
        """Get the variable for specified slot."""
        return self._tls._slot_variables.get(layer_name, {}).get(
            slot_name, None
        )

    def _get_embedding_variable(self, layer_name):
        """Get the variable for the specified ElasticDL embedding layer."""
        return self._tls._embed_variables.get(layer_name, None)

    def _create_slot_variable(self, layer_name, slot_name, initial_value):
        """Creates a slot variable in TensorFlow optimizer using given
        value.
        """
        embed_var = self._get_embedding_variable(layer_name)
        if embed_var is None:
            raise RuntimeError(
                "Embedding variable for layer %s should be already created."
                % (layer_name)
            )
        slot_var = self._create_slot_variable_in_optimizer(
            embed_var, slot_name, initial_value.shape, initial_value
        )
        slot_variables_dict = self._tls._slot_variables.setdefault(
            layer_name, {}
        )
        slot_variables_dict[slot_name] = slot_var
        return slot_var

    # This is a function modified from TensorFlow optimizers.
    # https://github.com/tensorflow/tensorflow/blob/
    # 69b1feac62276edcc509ac88af229c6236e645fe/tensorflow/python
    # /keras/optimizer_v2/optimizer_v2.py#L567
    def _create_slot_variable_in_optimizer(
        self, embed_var, slot_name, shape, initial_value
    ):
        """Create variable for a slot and save it in TensorFlow optimizer."""
        if slot_name not in self._opt._slot_names:
            self._opt._slot_names.append(slot_name)
        var_key = _var_key(embed_var)
        slot_dict = self._opt._slots.setdefault(var_key, {})
        slot_var = slot_dict.get(slot_name, None)
        if slot_var is None:
            slot_var_name = "%s/%s" % (embed_var._shared_name, slot_name)
            slot_var = self._opt.add_weight(
                name=slot_var_name,
                shape=shape,
                dtype=embed_var.dtype,
                initializer=initial_value,
                trainable=False,
            )
            slot_dict[slot_name] = slot_var
            self._opt._weights.append(slot_var)
            return slot_var
        else:
            raise RuntimeError(
                "Variable with var_key %s and slot_name %s is not expected to "
                "be in self._opt." % (var_key, slot_name)
            )

    def _update_embedding_param(self):
        """Report updated embedding vectors and slots to kv store."""
        for layer, ids in self._tls._unique_ids_all_layers.items():
            value = self._get_embedding_variable(layer).numpy()
            self._update_embedding_func(layer, ids, value)

            for slot in self._allowed_slot_names:
                value = self._get_slot_variable(layer, slot).numpy()
                slot_table_name = get_slot_table_name(layer, slot)
                self._update_embedding_func(slot_table_name, ids, value)

    def _delete_variables(self):
        # Slot variable access in optimizer requires corresponding embedding
        # variable information. Delete slot variables first.
        for layer_name, slots in self._tls._slot_variables.items():
            embed_var = self._get_embedding_variable(layer_name)
            embed_var_key = _var_key(embed_var)
            del self._opt._slots[embed_var_key]
            for _, var in slots.items():
                opt_weight_iter = 0
                while opt_weight_iter < len(self._opt._weights):
                    if var is self._opt._weights[opt_weight_iter]:
                        self._opt._weights.pop(opt_weight_iter)
                        break
                    else:
                        opt_weight_iter += 1
        for key in list(self._tls._slot_variables.keys()):
            del self._tls._slot_variables[key]

        # Delete variables in embed_variables.
        for key in list(self._tls._embed_variables.keys()):
            del self._tls._embed_variables[key]

        # Delete variables in unique_ids_all_layers.
        for key in list(self._tls._unique_ids_all_layers.keys()):
            del self._tls._unique_ids_all_layers[key]

    @property
    def allowed_slot_names(self):
        return self._allowed_slot_names

    # TODO(yunjian.lmh): Do not need to save slot_initial_value in
    #     optimizer wrapper after we do not need to support Redis.
    @property
    def slot_initial_value(self):
        return self._slot_initial_value
