import itertools
from multiprocessing import Process, Queue


def worker_loop(records, index_q, result_q, transform_fn):
    while True:
        index = index_q.get()
        if index is None:
            break
        record = records[index]
        res = transform_fn(record)
        result_q.put((index, res))


class ParallelTransform(object):
    def __init__(self, records, num_workers, transform_fn):
        self._records = records
        self._num_workers = num_workers
        self._transform_fn = transform_fn

        self._result_queue = Queue()
        self._index_queues = []
        self._workers = []

        self._worker_id_cycle = itertools.cycle(range(self._num_workers))
        self._put_record_it = iter(range(len(self._records)))
        self._get_record_it = iter(range(len(self._records)))
        self._cache_dict = {}

        for i in range(self._num_workers):
            index_queue = Queue()
            self._index_queues.append(index_queue)
            p = Process(
                target=worker_loop,
                args=(
                    self._records,
                    self._index_queues[i],
                    self._result_queue,
                    self._transform_fn,
                ),
            )
            p.daemon = True
            p.start()
            self._workers.append(p)

        for i in range(2 * self._num_workers):
            self._put_index()

    def _next_index(self):
        try:
            idx = next(self._put_record_it)
            return idx
        except StopIteration:
            return None

    def _put_index(self):
        worker_id = next(self._worker_id_cycle)
        idx = self._next_index()
        if idx is not None:
            self._index_queues[worker_id].put(idx)

    def next_data(self):
        try:
            index = next(self._get_record_it)
        except StopIteration:
            for i, w in enumerate(self._workers):
                self._index_queues[i].put(None)
                w.join()
            return None
        while True:
            data = self._cache_dict.pop(index, None)
            if data:
                self._put_index()
                return data

            data_id, data = self._result_queue.get()
            if data_id == index:
                self._put_index()
                return data
            else:
                self._cache_dict[data_id] = data
