"""Optimizer Wrapper for ElasticDL"""
# TODO(yunjian.lmh): move this file to PS module after we don't need to
#     support ps in master.


import threading

import numpy as np
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
from elasticdl.python.elasticdl.layers.embedding import Embedding


def _parse_lookup_values(values, key_index):
    """Parse looked up values recursively.

    This function parses looked up values from Redis recursively.
    For example, if `key_index` = `{
        layer_1: {slot_1: (0, 3), slot_2: (3, 6)},
        layer_2: {slot_1: (6, 12), slot_2: (12, 18)},
    }`,
    this function returns a python dictionary `{
        layer_1: {slot_1: values[0:3], slot_2: values[3:6]},
        layer_2: {slot_1: (6, 12), slot_2: (12, 18)},
    }`

    Arguments:
        values: A list of 1D `numpy.ndarray`.
        key_index: A dictionary of key index.

    Returns:
        A python dictionary of parsed values.

    """
    parsed_values = {}
    for k, v in key_index.items():
        if isinstance(v, dict):
            parsed_values[k] = _parse_lookup_values(values, v)
        else:
            start, end = v
            parsed_values[k] = np.concatenate(values[start:end]).reshape(
                end - start, -1
            )
    return parsed_values


def _get_embedding_layer_name_from_var(var):
    """Get name for ElasticDL embedding layer from variable."""
    # Assumes that for ElasticDL embedding layer, variable will be a
    # string representing its layer name
    if isinstance(var, str):
        return var
    return None


def _get_embedding_layer_name_from_key(key):
    """Get name for ElasticDL embedding layer from kv store key."""
    return "-".join(key.split("-")[:-2])


def _get_slot_name_from_key(key):
    """Get slot name from kv store key."""
    return key.split("-")[-2]


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
        kv_store_endpoint,
        embedding_dims,
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
        self._kv_store_endpoint = kv_store_endpoint
        self._embed_dims = embedding_dims
        self._use_async = use_async
        self._lookup_embedding_func = lookup_embedding_func
        self._update_embedding_func = update_embedding_func
        self._slot_initial_value = {}

        self._opt_weights_delete_lock = threading.Lock()
        self._tls = threading.local()
        self._init_thread_local()

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

        Arguments:
            grads_and_vars: A list of (gradient, variable) pairs.

        """
        # TODO (#1255): Discuss whether `OptimizerWrapper` needs a lock after
        # implementing PS.
        self._init_thread_local()

        grads_and_vars = list(grads_and_vars)

        # split `grads_and_vars` according to whether it is from
        # ElasticDL embedding layer
        grads_and_vars_local = []
        grads_and_vars_kv_store = []
        for grad, var in grads_and_vars:
            layer_name = _get_embedding_layer_name_from_var(var)
            if layer_name:
                grads_and_vars_kv_store.append((grad, layer_name))
            else:
                grads_and_vars_local.append((grad, var))

        # `_lookup_embeddings_and_slots` will raise Error if there are
        # unknown embedding keys
        embed_values, slot_values = self._lookup_embeddings_and_slots(
            grads_and_vars_kv_store
        )

        self._set_embedding_values_to_variables(
            grads_and_vars_kv_store, embed_values
        )
        self._set_slot_values_to_variables(slot_values)

        self._opt.apply_gradients(
            grads_and_vars_local + grads_and_vars_kv_store
        )

        self._report_to_kv_store()

        self._delete_variables()

    def _lookup_embeddings_and_slots(self, grads_and_vars):
        """Look up embedding vectors and slot values form kv store.

        This function looks up embedding vectors and slot values.
        It initializes unknown slot if exist.

        Arguments:
            grads_and_vars: A list of (gradient, layer name) pairs.

        Returns:
            A tuple of (`embedding_values`, `slot_values`). `embedding_values`
            is a python dictionary of {layer name: `embedding_vectors`} where
            `embedding_vectors` is a 2D `numpy.ndarray`. `slot_values` is a
            python dictionary of {layer name: {slot name: `slot_values`}}
            where `slot_values` is a 2D `numpy.ndarray`.

        Raises:
            RuntimeError: If any unknown embedding key exists.
        """

        arr = self._generate_lookup_keys(grads_and_vars)
        embed_keys, slot_keys, embed_key_index, slot_key_index = arr

        keys = embed_keys + slot_keys
        embed_keys_num = len(embed_keys)
        if self._lookup_embedding_func:
            values, unknown_keys = self._lookup_embedding_func(keys)

        if unknown_keys:
            # raise Error if an unknown embedding key exists
            if unknown_keys[0] < embed_keys_num:
                raise RuntimeError(
                    "Failed to get key %s from kv store."
                    % embed_keys[unknown_keys[0]]
                )

            # initialize unknown slots
            for idx in unknown_keys:
                key = keys[idx]
                layer_name = _get_embedding_layer_name_from_key(key)
                slot_name = _get_slot_name_from_key(key)
                values[idx] = self._initialize_unknown_slot(
                    layer_name, slot_name
                )

        embed_values = _parse_lookup_values(
            values[:embed_keys_num], embed_key_index
        )
        slot_values = _parse_lookup_values(
            values[embed_keys_num:], slot_key_index
        )
        return embed_values, slot_values

    def _generate_lookup_keys(self, grads_and_vars):
        """Generate lookup keys from a list of (gradient, variable) pairs.

        Arguments:
            grads_and_vars: A list of (gradient, layer name) pairs.

        Returns:
            A tuple of (`embedding_keys`, `slot_keys`, `embedding_key_index`,
                `slot_key_index`).
            `embedding_keys`: A list of keys for embedding vectors in kv
                store.
            `slot_keys`: A list of keys for slots in kv store.
            `embedding_key_index`: A python dictionary records the position
                of embedding keys for the same layer, i.e. an item
                `{layer_name: (start, end)}` means `embedding_keys[start:end]`
                are keys for the same layer named `layer_name`.
            `slot_key_index`: A python dictionary records the position of slot
                keys for the same layer and the smae slot, i.e. an item
                `{layer_name: {slot_name: (start, end)}}` means
                `slot_keys[start:end]` are keys for the same layer named
                `layer_name` and same slot named `slot_name`.

        """
        embed_keys = []
        embed_key_index = {}
        slot_keys = []
        slot_key_index = {}
        self._tls._unique_ids_all_layers = {}

        # generate keys
        for it, (grad, layer_name) in enumerate(grads_and_vars):
            # de-duplicate gradient's indices
            unique_ids, indices = tf.unique(grad.indices)
            unique_ids = unique_ids.numpy()
            if layer_name in self._tls._unique_ids_all_layers:
                # TODO: support grads_and_vars with duplicated layer name
                logger.warning(
                    "grads_and_vars has duplicated layer name %s." % layer_name
                )
            self._tls._unique_ids_all_layers[layer_name] = unique_ids
            grad_new = tf.IndexedSlices(grad.values, indices)
            grads_and_vars[it] = (grad_new, layer_name)

            # generate embedding keys
            start = len(embed_keys)
            embed_keys.extend(
                [Embedding.get_key([layer_name, i]) for i in unique_ids]
            )
            end = len(embed_keys)
            embed_key_index[layer_name] = (start, end)

            # generate slot keys
            for slot in self._allowed_slot_names:
                start = len(slot_keys)
                slot_keys.extend(
                    [
                        Embedding.get_key([layer_name, slot, i])
                        for i in unique_ids
                    ]
                )
                end = len(slot_keys)
                slot_key_index.setdefault(layer_name, {}).setdefault(
                    slot, (start, end)
                )
        return embed_keys, slot_keys, embed_key_index, slot_key_index

    def _initialize_unknown_slot(self, layer_name, slot_name):
        """Initialize unknown slot."""
        slot_dim = self._embed_dims[layer_name]
        initial_value = self._slot_initial_value[slot_name]
        return np.full((slot_dim,), initial_value, np.float32)

    def _set_embedding_values_to_variables(self, grads_and_vars, values):
        """Set embedding values to embedding variables."""
        for i, (grad, layer_name) in enumerate(grads_and_vars):
            value = values[layer_name]
            variable = self._get_embedding_variable(layer_name)
            if variable is None:
                variable = self._create_embedding_variable(layer_name, value)
            else:
                variable.assign(value)
            grads_and_vars[i] = (grad, variable)

    def _set_slot_values_to_variables(self, values):
        """Set slot values to slot variables in TensorFlow optimizers."""
        for layer_name, slots in values.items():
            for slot_name, slot_value in slots.items():
                # `variable` points to the variable object saved in
                # TensorFlow optimizer, i.e. self._opt
                variable = self._get_slot_variable(layer_name, slot_name)
                if variable is None:
                    self._create_slot_variable(
                        layer_name, slot_name, slot_value
                    )
                else:
                    variable.assign(slot_value)

    def _get_slot_variable(self, layer_name, slot_name):
        """Get the variable for specified slot."""
        return self._tls._slot_variables.get(layer_name, {}).get(
            slot_name, None
        )

    def _get_embedding_variable(self, layer_name):
        """Get the variable for the specified ElasticDL embedding layer."""
        return self._tls._embed_variables.get(layer_name, None)

    # TODO: refactor _create_slot_variable and _create_embedding_variable
    # into one function
    def _create_embedding_variable(self, layer_name, initial_value):
        """Create a variable for an ElasticDL embedding layer."""
        dim = self._embed_dims[layer_name]
        # Use shape `(None, dim)` for embedding variable because `shape[0]`
        # equals to the number of unique ids in the minibatch data, and
        # this number may differ between different iterations
        shape = tf.TensorShape((None, dim))

        if self._tls._embed_variables.get(layer_name, None) is not None:
            raise RuntimeError(
                "Embedding variable with layer name=%s has already be "
                "created." % (layer_name)
            )

        embed_var = tf.Variable(
            initial_value,
            name=layer_name + str(threading.get_ident()),
            shape=shape,
            dtype=tf.float32,
            trainable=False,
        )
        self._tls._embed_variables[layer_name] = embed_var
        return embed_var

    def _create_slot_variable(self, layer_name, slot_name, initial_value):
        """Create a variable for the specified slot."""
        dim = self._embed_dims[layer_name]
        # Use shape `(None, dim)` for slot variable because `shape[0]`
        # equals to the number of unique ids in the minibatch data, and
        # this number may differ between different iterations
        shape = tf.TensorShape((None, dim))

        slot_variables_dict = self._tls._slot_variables.setdefault(
            layer_name, {}
        )
        if slot_variables_dict.get(slot_name, None) is not None:
            raise RuntimeError(
                "Slot variable with (layer name=%s, slot name=%s) has "
                "already be created." % (layer_name, slot_name)
            )

        embed_var = self._get_embedding_variable(layer_name)
        if embed_var is None:
            raise RuntimeError(
                "Embedding variable for layer %s should be already created."
                % (layer_name)
            )
        slot_var = self._create_slot_variable_in_optimizer(
            embed_var, slot_name, shape, initial_value
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

    def _report_to_kv_store(self):
        """Report updated embedding vectors and slots to kv store."""
        keys = []
        values = []
        for layer, ids in self._tls._unique_ids_all_layers.items():
            value = self._get_embedding_variable(layer).numpy()
            for id, v in zip(ids, value):
                keys.append(Embedding.get_key([layer, id]))
                values.append(v)

            for slot in self._allowed_slot_names:
                value = self._get_slot_variable(layer, slot).numpy()
                for id, v in zip(ids, value):
                    keys.append(Embedding.get_key([layer, slot, id]))
                    values.append(v)

        if self._update_embedding_func:
            self._update_embedding_func(keys, values)

    def _delete_variables(self):
        # Slot variable access in optimizer requires corresponding embedding
        # variable information. Delete slot variables first.
        for layer_name, slots in self._tls._slot_variables.items():
            embed_var = self._get_embedding_variable(layer_name)
            embed_var_key = _var_key(embed_var)
            del self._opt._slots[embed_var_key]
            for _, var in slots.items():
                opt_weight_iter = 0
                with self._opt_weights_delete_lock:
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
