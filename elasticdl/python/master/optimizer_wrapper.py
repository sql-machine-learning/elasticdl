"""Optimizer Wrapper for ElasticDL"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam

from elasticdl.python.common.embedding_service import EmbeddingService
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


class OptimizerWrapper(object):
    """ ElasticDL optimizer wrapper.

    If model does not use ElasticDL embedding layer, `OptimizerWrapper`
    does nothing but calls `apply_gradients` function of TensorFlow optimizer.
    Otherwise, `OptimizerWrapper` looks up embedding vectors and slot values
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
        self._slot_initial_value = {}

        # TODO: support more TensorFlow optimizers
        # "-" in slot name is not supported
        if isinstance(opt, SGD):
            self._allowed_slot_names = []
            if opt._momentum:
                self._allowed_slot_names.append("momentum")
            for slot in self._allowed_slot_names:
                self._slot_initial_value[slot] = 0.0

        elif isinstance(opt, Adam):
            self._allowed_slot_names = ["m", "v"]
            if self._opt.amsgrad:
                self._allowed_slot_names.append("vhat")
            for slot in self._allowed_slot_names:
                self._slot_initial_value[slot] = 0.0
        else:
            raise NotImplementedError(
                "Optimizer %s is not supported in ElasticDL." % type(opt)
            )

        # record unique ids of gradients
        self._unique_ids_all_layers = {}

    def apply_gradients(self, grads_and_vars):
        """Update variable values.

        Arguments:
            grads_and_vars: A list of (gradient, variable) pairs.

        """
        grads_and_vars = list(grads_and_vars)

        # split `grads_and_vars` according to whether it is from
        # ElasticDL embedding layer
        grads_and_vars_local = []
        grads_and_vars_kv_store = []
        for grad, var in grads_and_vars:
            layer_name = _get_embedding_layer_name_from_var(grad, var)
            if layer_name:
                grads_and_vars_kv_store.append((grad, layer_name))
            else:
                grads_and_vars_local.append((grad, var))

        # `_lookup_embeddings_and_slots` will raise Error if there are
        # unknown embedding keys
        embed_values, slot_values = self._lookup_embeddings_and_slots(
            grads_and_vars_kv_store
        )

        # TODO: implement the following logic:
        # 1. set embedding values and slot values to TensorFlow Variables
        # 2. call self._opt.apply_gradients
        # 3. report updated values to Redis

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
        values, unknown_keys = EmbeddingService.lookup_embedding(
            keys=keys, embedding_service_endpoint=self._kv_store_endpoint
        )

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

        # generate keys
        for it, (grad, layer_name) in enumerate(grads_and_vars):
            # de-duplicate gradient's indices
            unique_ids, indices = tf.unique(grad.indices)
            unique_ids = unique_ids.numpy()
            self._unique_ids_all_layers[layer_name] = unique_ids
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
        slot_dim = self._embedding_dims[layer_name]
        initial_value = self._slot_initial_value[slot_name]
        return np.full((slot_dim,), initial_value, np.float32)

    @property
    def allowed_slot_names(self):
        return self._allowed_slot_names
