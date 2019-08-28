"""Mock Kv Store for unittest"""


class MockKvStore(object):
    def __init__(self, store={}):
        if store:
            for k, v in store.items():
                if len(v.shape) != 1:
                    raise ValueError(
                        "Value of key %s are expected to be one-dimension "
                        "tensor, received shape %s." % (k, str(v.shape))
                    )
        self._store = store

    def lookup(self, keys=[], *args, **kwargs):
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
        self, keys=[], values=[], set_if_not_exists=False, *args, **kwargs
    ):
        for key, value in zip(keys, values):
            if set_if_not_exists:
                self._store.setdefault(key, value)
            else:
                self._store[key] = value
