# No partition, all in the first ps
def NoPartition(v, ps_size):
    return [(0, v)]


# Partition v according to the name hashing.
def HashPartition(v, ps_size):
    if v is None:
        return [(i, None) for i in range(ps_size)]
    if len(v) == 0:
        raise ValueError('Empty input used in HashPartition.')
    if ps_size == 1:
        return [(0, v)]
    if type(v) == dict:
        # (name, data) dict from push
        results = [(i, {}) for i in range(ps_size)]
        for name, data in v.items():
            index = hash(name) % ps_size
            results[index][1][name] = data
    elif type(v) == list and type(v[0]) == str:
        # name list from pull
        results = [(i, []) for i in range(ps_size)]
        for name in v:
            index = hash(name) % ps_size
            results[index][1].append(name)
    else:
        raise TypeError('Illegal v type %s, only dict or '
                        'str list is supported.' % str(type(v)))
    return results


# PSClientComm defines the API interface with ps
class PSClientComm(object):
    def __init__(self):
        pass

    def push(self, base_step, sub_step, grads):
        pass

    def pull(self, names=None):
        pass


# Multi-thread implementation of PSClientComm
class MultiThreadPSClientComm(PSClientComm):
    def __init__(self, ps):
        super(MultiThreadPSClientComm, self).__init__()
        self._ps = ps

    def push(self, base_step, sub_step, grads):
        self._ps.push(base_step, sub_step, grads)

    def pull(self, min_step=0, names=None):
        return self._ps.pull(min_step=min_step, names=names)


# ParameterSererClient uses PSClientComm for ps data transfer.
class ParameterServerClient(object):
    def __init__(self,
                 ps_size=1,
                 ps_configs=None,
                 comm_class=MultiThreadPSClientComm,
                 partition_func=NoPartition):
        assert(ps_size > 0)
        assert(len(ps_configs) == ps_size)
        self._ps_size = ps_size
        self._partition_func = partition_func
        self._clients = [comm_class(ps_configs[i])
                         for i in range(self._ps_size)]
        self._base_step = [0 for _ in range(self._ps_size)]

    def push(self, sub_step=0, grads=None):
        partition_result = self._partition_func(grads, self._ps_size)
        # TODO: multithread optimization, one thread per ps communication.
        for index, g in partition_result:
            if len(g) > 0:
                self._clients[index].push(self._base_step[index],
                                          sub_step, g)

    def pull(self, min_step=0, names=None):
        pull_result = {}
        partition_result = self._partition_func(names, self._ps_size)
        # TODO: multithread optimization, one thread per ps communication.
        for index, n in partition_result:
            # TODO: rewrite the logic after ps.pull supports min_step
            while True:
                ps_step, ps_vars = self._clients[index].pull(names=n)
                if ps_step >= min_step:
                    break
            self._base_step[index] = ps_step
            pull_result.update(ps_vars)
        return self.get_min_base_step(), pull_result

    def get_min_base_step(self):
        return min(self._base_step)
