import numpy as np
import tensorflow as tf

from elasticdl.python.common.embedding_service import EmbeddingService
from elasticdl.python.elasticdl.layers.embedding import Embedding


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

        # TODO: support more TensorFlow optimizers
        if isinstance(opt, tf.keras.optimizers.SGD):
            self._allowed_slot_names = []
            if opt._momentum:
                self._allowed_slot_names.append("momentum")
        elif isinstance(opt, tf.keras.optimizers.Adam):
            self._allowed_slot_names = ["m", "v"]
            if self._opt.amsgrad:
                self._allowed_slot_names.append("vhat")
        else:
            raise NotImplementedError(
                "Optimizer %s is not supported in ElasticDL." % type(opt)
            )

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
            layer_name = self._get_embedding_layer_name(grad, var)
            if layer_name:
                grads_and_vars_kv_store.append((grad, layer_name))
            else:
                grads_and_vars_local.append((grad, var))

        # `_lookup_embeddings_and_slots` will raise Error if appears
        # unknown embedding keys
        embedding_values, slot_values = self._lookup_embeddings_and_slots(
            grads_and_vars_kv_store
        )

        # TODO: implement the following logic to do model updating:
        # 1. set embedding values and slot values to TensorFlow Variables
        # 2. call self._opt.apply_gradients
        # 3. report updated values to Redis

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

    def _get_embedding_layer_name(self, grad, var):
        """Return layer name for ElasticDL embedding layer."""
        # Assumes that for ElasticDL embedding layer, variable will be a
        # string representing its layer name
        if isinstance(var, str):
            return var
        return None

    def _initialize_unknown_slot(self, layer_name):
        """Initialize unknown slot."""
        slot_dim = self._embedding_dims[layer_name]
        if isinstance(self._opt, tf.keras.optimizers.Adam) or isinstance(
            self._opt, tf.keras.optimizers.SGD
        ):
            return np.zeros(slot_dim, np.float32)
        else:
            raise NotImplementedError(
                "Optimizer %s is not supported in ElasticDL." % type(self._opt)
            )

    @property
    def allowed_slot_names(self):
        return self._allowed_slot_names
