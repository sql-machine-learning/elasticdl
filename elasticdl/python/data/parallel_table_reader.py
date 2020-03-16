from multiprocessing import Process, Queue

from odps import ODPS


class ParallelTableReader(object):
    def __init__(
        self,
        access_id,
        access_key,
        project,
        endpoint,
        table,
        partition,
        columns,
        batch_size,
        num_parallel_processes,
        transform_fn,
    ):
        self._odps_table = ODPS(
            access_id, access_key, project, endpoint
        ).get_table(table)
        self._partition = partition
        self._columns = columns
        self._batch_size = batch_size

        self._num_parallel_processes = num_parallel_processes
        self._transform_fn = transform_fn

        self._result_queue = Queue()
        self._index_queues = []
        self._workers = []

        self._shards = []
        self._shard_idx = 0
        self._worker_idx = 0

        for i in range(self._num_parallel_processes):
            index_queue = Queue()
            self._index_queues.append(index_queue)

            p = Process(target=self._worker_loop, args=(i,))
            p.daemon = True
            p.start()
            self._workers.append(p)

    def reset(self, shards):
        self._create_shards(shards)
        self._shard_idx = 0
        for i in range(2 * self._num_parallel_processes):
            self._put_index()

    def get_shards_count(self):
        return len(self._shards)

    def get_records(self):
        data = self._result_queue.get()
        self._put_index()
        return data

    def stop(self):
        for q in self._index_queues:
            q.put((None, None))

    def _worker_loop(self, worker_id):
        while True:
            index = self._index_queues[worker_id].get()
            if index[0] is None and index[1] is None:
                break
            if self._columns is None:
                self._columns = self._odps_table.schema.names
            records = []
            with self._odps_table.open_reader(
                partition=self._partition, reopen=False
            ) as reader:
                for record in reader.read(
                    start=index[0], count=index[1], columns=self._columns
                ):
                    record = [str(record[column]) for column in self._columns]
                    record = self._transform_fn(record)
                    records.append(record)
            self._result_queue.put(records)

    def _create_shards(self, shards):
        start = shards[0]
        count = shards[1]
        m = count // self._batch_size
        n = count % self._batch_size

        for i in range(m):
            self._shards.append(
                (start + i * self._batch_size, self._batch_size)
            )
        if n != 0:
            self._shards.append((start + m * self._batch_size, n))

    def _next_worker_id(self):
        cur_id = self._worker_idx
        self._worker_idx += 1
        if self._worker_idx == self._num_parallel_processes:
            self._worker_idx = 0
        return cur_id

    def _put_index(self):
        # put index to the index queue of each worker
        # with Round-Robin way
        if self._shard_idx < len(self._shards):
            worker_id = self._next_worker_id()
            shard = self._shards[self._shard_idx]
            self._index_queues[worker_id].put(shard)
            self._shard_idx += 1
