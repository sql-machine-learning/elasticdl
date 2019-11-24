"""Mock Kv Store for unittest"""
import numpy as np


class MockKvStoreV2(object):
    """A Kv Store for unittest."""

    def __init__(self, store={}):
        if store:
            for k, v in store.items():
                if len(v.shape) != 1:
                    raise ValueError(
                        "Value of key %s are expected to be one-dimension "
                        "tensor, received shape %s." % (k, str(v.shape))
                    )
        self._store = store

    def lookup(
        self, keys=[], embedding_service_endpoint=None, parse_type=np.float32
    ):
        """
        Lookup values in Kv Store. Arguments should keep same with
        EmbeddingService.lookup_embedding.
        """
        values = []
        unknown_key = []
        for i, key in enumerate(keys):
            if key in self._store:
                values.append(self._store.get(key))
            else:
                values.append(None)
                unknown_key.append(i)
        return values, unknown_key

    def update(
        self,
        keys=[],
        embedding_vectors=[],
        embedding_service_endpoint=None,
        set_if_not_exist=False,
    ):
        """
        Update values in Kv Store. Arguemnts should keep same with
        EmbeddingService.update_embedding.
        """
        for key, value in zip(keys, embedding_vectors):
            if set_if_not_exist:
                self._store.setdefault(key, value)
            else:
                self._store[key] = value


# TODO(yunjian.lmh): MockKvStore can be removed when we do not support Redis
class MockKvStore(object):
    """A Kv Store for unittest."""

    def __init__(self, store={}):
        if store:
            for k, v in store.items():
                if len(v.shape) != 1:
                    raise ValueError(
                        "Value of key %s are expected to be one-dimension "
                        "tensor, received shape %s." % (k, str(v.shape))
                    )
        self._store = store

    def lookup(
        self, keys=[], embedding_service_endpoint=None, parse_type=np.float32
    ):
        """
        Lookup values in Kv Store. Arguments should keep same with
        EmbeddingService.lookup_embedding.
        """
        values = []
        unknown_key = []
        for i, key in enumerate(keys):
            if key in self._store:
                values.append(self._store.get(key))
            else:
                values.append(None)
                unknown_key.append(i)
        return values, unknown_key

    def update(
        self,
        keys=[],
        embedding_vectors=[],
        embedding_service_endpoint=None,
        set_if_not_exist=False,
    ):
        """
        Update values in Kv Store. Arguemnts should keep same with
        EmbeddingService.update_embedding.
        """
        for key, value in zip(keys, embedding_vectors):
            if set_if_not_exist:
                self._store.setdefault(key, value)
            else:
                self._store[key] = value
