from binascii import crc32


# No partition, all in the first ps
def no_partition(v, _):
    return [v]


# Partition v according to the name hashing.
def hash_partition(v, ps_size):
    # a simple stable string hash
    def _hash(s):
        return crc32(s.encode())

    if v is None:
        return [None for i in range(ps_size)]
    if not v:
        raise ValueError("Empty input used in HashPartition.")
    if ps_size == 1:
        return [v]
    if isinstance(v, dict):
        # (name, data) dict from push
        results = [{} for i in range(ps_size)]
        for name, data in v.items():
            index = _hash(name) % ps_size
            results[index][name] = data
    elif isinstance(v, list):
        # name list from pull
        results = [[] for i in range(ps_size)]
        for name in v:
            index = _hash(name) % ps_size
            results[index].append(name)
    else:
        raise TypeError(
            "Illegal v type %s, only dict or "
            "str list is supported." % str(type(v))
        )
    return results


# Multi-thread implementation of PSClientComm
class MultiThreadPSClientComm(object):
    def __init__(self, ps):
        self._ps = ps

    def push(self, base_step, sub_step, grads):
        if grads:
            self._ps.push(base_step, sub_step, grads)

    def pull(self, names=None, min_step=0):
        return self._ps.pull(names=names, min_step=min_step)


# ParameterSererClient uses PSClientComm for ps data transfer.
class ParameterServerClient(object):
    def __init__(
        self,
        ps_configs=None,
        comm_class=MultiThreadPSClientComm,
        partition_func=no_partition,
    ):
        self._ps_size = (
            1 if partition_func == no_partition else len(ps_configs)
        )
        self._partition_func = partition_func
        self._clients = [
            comm_class(ps_configs[i]) for i in range(self._ps_size)
        ]
        self._base_step = [0 for _ in range(self._ps_size)]

    def push(self, sub_step=0, grads=None):
        partition_result = self._partition_func(grads, self._ps_size)
        # TODO: multithread optimization, one thread per ps communication.
        for index, g in enumerate(partition_result):
            self._clients[index].push(self._base_step[index], sub_step, g)

    def pull(self, min_step=0, names=None):
        pull_result = {}
        partition_result = self._partition_func(names, self._ps_size)
        # TODO: multithread optimization, one thread per ps communication.
        for index, n in enumerate(partition_result):
            ps_step, ps_vars = self._clients[index].pull(
                names=n, min_step=min_step
            )
            self._base_step[index] = ps_step
            pull_result.update(ps_vars)
        return self.get_min_base_step(), pull_result

    def get_min_base_step(self):
        return min(self._base_step)
