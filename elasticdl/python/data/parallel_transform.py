import itertools
from multiprocessing import Process, Queue


def worker_loop(records, index_q, result_q, transform_fn):
    """
    The worker loop gets a index from the index queue, and
    transform the records[index], then the result is put to
    the result queue.

    If a None index is got, the loop finishes.

    :param records: a list object contains many records waiting to
                    be transformed
    :param index_q: a queue which generates the record index in records
    :param result_q: a queue which stores the transform results
    :param transform_fn: the transform function
    """
    while True:
        index = index_q.get()
        if index is None:
            break
        record = records[index]
        res = transform_fn(record)
        result_q.put((index, res))


class ParallelTransform(object):
    """
    The ParallelTransform applies transform_fn to records with a number
    of workers with low latency.

    Worker process side:
    1. get index from index queue
    2. transform_fn(records[index])
    3. put result to result queue

    Main process side:
    1. put index to index queue of workers in round-robin way
    2. get transformed data from result queue

    We want to achieve two things:
    1. Data samples in input records are transformed with low latency.
       We transform the data samples one by one. Once we call `next_data`
       method to get a transformed result, the next index is put to a
       index queue and triggers a worker process.

    2. The output order is strictly the same with the input order.
       We use a `cache_dict` to adjust the order, since the result order
       getting from result queue is sequential sometimes.
    """

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

        # prime the prefetch loop
        for i in range(2 * self._num_workers):
            self._put_index()

    def _next_index(self):
        try:
            idx = next(self._put_record_it)
            return idx
        except StopIteration:
            return None

    def _put_index(self):
        # put index to the index queue of each worker
        # with Round-Robin way
        worker_id = next(self._worker_id_cycle)
        idx = self._next_index()
        if idx is not None:
            self._index_queues[worker_id].put(idx)

    def next_data(self):
        # get data one-by-one, and the output order is strictly
        # the same with the input order.
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
