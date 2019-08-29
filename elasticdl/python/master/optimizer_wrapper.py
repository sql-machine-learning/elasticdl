"""Optimizer Wrapper for ElasticDL"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras import backend

from elasticdl.python.common.embedding_service import EmbeddingService
from elasticdl.python.elasticdl.layers.embedding import Embedding


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
    Otherwise, `OptimizerWrapper` lookups embedding vectors and slot values
    from external kv store before updating variables, and updates embedding
    vectors and slot values in kv store after updating variables.
    """

    def __init__(self, opt, kv_store_endpoint, embedding_dims):
        """
        Arguments:
            opt: A TensorFlow optimizer instance.
            kv_store_endpoint: The endpoint to kv store.
            embedding_dims: A python dictionary of
                {layer name: `embedding_dim`} where layer name is the
                name of ElasticDL embedding layer and `embedding_dim`
                is the output dimension of corresponding embedding layer.
        """
        self._opt = opt
        self._kv_store_endpoint = kv_store_endpoint
        self._embedding_dims = embedding_dims
        self._embedding_variables = {}
        self._slot_variables = {}

        # TODO: support more TensorFlow optimizers
        if isinstance(opt, SGD):
            self._allowed_slot_names = []
            if opt._momentum:
                self._allowed_slot_names.append("momentum")
        elif isinstance(opt, Adam):
            self._allowed_slot_names = ["m", "v"]
            if self._opt.amsgrad:
                self._allowed_slot_names.append("vhat")
        else:
            raise NotImplementedError(
                "Optimizer %s is not supported in ElasticDL." % type(opt)
            )

        self._unique_ids_all_layers = {}

    def apply_gradients(self, grads_and_vars):
        """Update variable values.

        Arguments:
            grads_and_vars: A list of (gradient, variable) pairs.

        Returns:
            None.
        """
        if not isinstance(grads_and_vars, list):
            grads_and_vars = list(grads_and_vars)

        # split `grads_and_vars` according to whether it is from
        # ElasticDL embedding layer
        grads_and_vars_local = []
        grads_and_vars_kv_store = []
        for grad, var in grads_and_vars:
            layer_name = self._get_embedding_layer_name_from_grad_var(
                grad, var
            )
            if layer_name:
                grads_and_vars_kv_store.append((grad, layer_name))
            else:
                grads_and_vars_local.append((grad, var))

        # `_lookup_embeddings_and_slots` will raise Error if appears
        # unknown embedding keys
        embedding_values, slot_values = self._lookup_embeddings_and_slots(
            grads_and_vars_kv_store
        )

        self._set_embedding_values_to_variables(
            grads_and_vars_kv_store, embedding_values
        )
        self._set_slot_values_to_variables(slot_values)

        # TODO: implement the following logic to do model updating:
        # * call self._opt.apply_gradients
        # * report updated values to Redis

    def _lookup_embeddings_and_slots(self, grads_and_vars):
        """Lookup embedding vectors and slot values form kv store.

        This function lookups embedding vectors and slot values.
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
        # embedding keys to lookup in kv store
        embed_names = []
        # embed_layer_index = {layer_name: (start, end)} means
        # embed_names[start: end] are embedding keys for the same layer
        embed_layer_index = {}

        # slot keys to lookup in kv store
        slot_names = []
        # slot_layer_index = {layer_name: (start, end)} means
        # slot_names[start: end] are slot keys for the same layer
        slot_layer_index = {}

        # record unique ids of gradients
        unique_ids_all_layers = {}

        # generate keys
        for it, (grad, layer_name) in enumerate(grads_and_vars):
            # de-duplicate gradient's indices
            unique_ids, indices = tf.unique(grad.indices)
            unique_ids = unique_ids.numpy()
            unique_ids_all_layers[layer_name] = unique_ids
            grad_new = tf.IndexedSlices(grad.values, indices)
            grads_and_vars[it] = (grad_new, layer_name)

            # generate embedding keys
            embed_names_single_layer = [
                Embedding.get_key([layer_name, i]) for i in unique_ids
            ]
            embed_layer_index[layer_name] = (
                len(embed_names),
                len(embed_names) + len(embed_names_single_layer),
            )
            embed_names.extend(embed_names_single_layer)

            # generate slot keys
            slot_names_single_layer = [
                Embedding.get_key([layer_name, slot, i])
                for slot in self._allowed_slot_names
                for i in unique_ids
            ]
            slot_layer_index[layer_name] = (
                len(slot_names),
                len(slot_names) + len(slot_names_single_layer),
            )
            slot_names.extend(slot_names_single_layer)

        # lookup in EmbeddingService
        keys = embed_names + slot_names
        values, unknown_keys = EmbeddingService.lookup_embedding(
            keys=keys, embedding_service_endpoint=self._kv_store_endpoint
        )

        # raise Error if an unknown embedding key exists
        embed_keys_num = len(embed_names)
        if unknown_keys and unknown_keys[0] < embed_keys_num:
            raise RuntimeError(
                "Failed to get key %s from kv store."
                % embed_names[unknown_keys[0]]
            )

        # initialize unknown slots
        for idx in unknown_keys:
            layer_name = self._get_embedding_layer_name_from_key(keys[idx])
            values[idx] = self._initialize_unknown_slot(layer_name)

        # parse embedding vectors
        embedding_values = {}
        for layer_name, (start, end) in embed_layer_index.items():
            num = end - start
            embedding_values[layer_name] = np.concatenate(
                values[start:end], axis=0
            ).reshape(num, -1)

        # parse slot values
        slot_values = {}
        values = values[embed_keys_num:]
        for layer_name, (start, end) in slot_layer_index.items():
            num = end - start
            num_per_slot = num // len(self._allowed_slot_names)
            offset = start
            for slot_name in self._allowed_slot_names:
                left = offset
                right = offset + num_per_slot
                slot_values.setdefault(layer_name, {})[
                    slot_name
                ] = np.concatenate(values[left:right], axis=0).reshape(
                    num_per_slot, -1
                )
                offset = right

        self._unique_ids_all_layers = unique_ids_all_layers
        return embedding_values, slot_values

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
                # `variable` points point to the variable object saved in
                # TensorFlow optimizer, i.e. self._opt
                variable = self._get_slot_variable(layer_name, slot_name)
                if variable is None:
                    self._create_slot_variable(
                        layer_name, slot_name, slot_value
                    )
                else:
                    variable.assign(slot_value)

    def _get_embedding_layer_name_from_grad_var(self, grad, var):
        """Get name for ElasticDL embedding layer from `(grad, var)` pair."""
        # Assumes that for ElasticDL embedding layer, variable will be a
        # string representing its layer name
        if isinstance(var, str):
            return var
        return None

    def _get_embedding_layer_name_from_key(self, key):
        """Get name for ElasticDL embedding layer from kv store key."""
        return "-".join(key.split("-")[:-2])

    def _initialize_unknown_slot(self, layer_name):
        """Initialize unknown slot."""
        slot_dim = self._embedding_dims[layer_name]
        if isinstance(self._opt, (Adam, SGD)):
            return np.zeros(slot_dim, np.float32)
        else:
            raise NotImplementedError(
                "Optimizer %s is not supported in ElasticDL." % type(self._opt)
            )

    def _get_slot_variable(self, layer_name, slot_name):
        """Get the variable for specified slot."""
        return self._slot_variables.get(layer_name, {}).get(slot_name, None)

    def _get_embedding_variable(self, layer_name):
        """Get the variable for the specified ElasticDL embedding layer."""
        return self._embedding_variables.get(layer_name, None)

    def _create_embedding_variable(self, layer_name, initial_value=None):
        """Create a variable for an ElasticDL embedding layer."""
        dim = self._embedding_dims[layer_name]
        shape = tf.TensorShape((None, dim))
        if initial_value is None:
            initial_value = tf.zeros((1, dim))

        if self._embedding_variables.get(layer_name, None) is not None:
            raise RuntimeError(
                "Embedding variable with layer name=%s has already be "
                "created." % (layer_name)
            )

        embedding_var = tf.Variable(
            initial_value,
            name=layer_name,
            shape=shape,
            dtype=tf.float32,
            trainable=False,
        )
        self._embedding_variables[layer_name] = embedding_var
        return embedding_var

    def _create_slot_variable(self, layer_name, slot_name, initial_value=None):
        """Create a variable for the specified slot."""
        dim = self._embedding_dims[layer_name]
        shape = tf.TensorShape((None, dim))
        if initial_value is None:
            initial_value = tf.zeros((1, dim))

        slot_variables_dict = self._slot_variables.setdefault(layer_name, {})
        if slot_variables_dict.get(slot_name, None) is not None:
            raise RuntimeError(
                "Slot variable with (layer name=%s, slot name=%s) has "
                "already be created." % (layer_name, slot_name)
            )

        embedding_var = self._get_embedding_variable(layer_name)
        if embedding_var is None:
            embedding_var = self._create_embedding_variable(layer_name)
        slot_var = self._create_slot_variable_in_optimizer(
            embedding_var, slot_name, shape, initial_value
        )
        slot_variables_dict[slot_name] = slot_var
        return slot_var

    # This is a function modified from TensorFlow optimizers.
    # https://github.com/tensorflow/tensorflow/blob/
    # 69b1feac62276edcc509ac88af229c6236e645fe/tensorflow/python
    # /keras/optimizer_v2/optimizer_v2.py#L567
    def _create_slot_variable_in_optimizer(
        self, embedding_var, slot_name, shape, initial_value
    ):
        """Create variable for a slot and save it in TensorFlow optimizer."""
        if slot_name not in self._opt._slot_names:
            self._opt._slot_names.append(slot_name)
        var_key = _var_key(embedding_var)
        slot_dict = self._opt._slots.setdefault(var_key, {})
        slot_var = slot_dict.get(slot_name, None)
        if slot_var is None:
            slot_var_name = "%s/%s/%s" % (
                self._opt._name,
                embedding_var._shared_name,
                slot_name,
            )
            slot_var = tf.Variable(
                name=slot_var_name,
                dtype=embedding_var.dtype,
                trainable=False,
                shape=shape,
                initial_value=initial_value,
            )
            backend.track_variable(slot_var)
            slot_dict[slot_name] = slot_var
            self._opt._weights.append(slot_var)
            return slot_var
        else:
            raise RuntimeError(
                "Variable with var_key %s and slot_name %s is not expected to "
                "be in self._opt." % (var_key, slot_name)
            )

    @property
    def allowed_slot_names(self):
        return self._allowed_slot_names
