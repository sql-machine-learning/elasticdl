import os
import sys
import time
from multiprocessing import Process, Queue

import odps
from odps import ODPS
from odps.models import Schema

from elasticdl.python.common.constants import MaxComputeConfig
from elasticdl.python.common.log_utils import default_logger as logger


def _nested_list_size(l):
    """
    Obtains the memory size for the nested list.
    """
    total = sys.getsizeof(l)
    for i in l:
        if isinstance(i, list):
            total += _nested_list_size(i)
        else:
            total += sys.getsizeof(i)

    return total


def _configure_odps_options(endpoint, options=None):
    if endpoint is not None and endpoint != "":
        odps.options.retry_times = options.get("odps.options.retry_times", 5)
        odps.options.read_timeout = options.get(
            "odps.options.read_timeout", 200
        )
        odps.options.connect_timeout = options.get(
            "odps.options.connect_timeout", 200
        )
        odps.options.tunnel.endpoint = options.get(
            "odps.options.tunnel.endpoint", None
        )
        if (
            odps.options.tunnel.endpoint is None
            and "service.odps.aliyun-inc.com/api" in endpoint
        ):
            odps.options.tunnel.endpoint = "http://dt.odps.aliyun-inc.com"


def is_odps_configured():
    return all(
        k in os.environ
        for k in (
            MaxComputeConfig.PROJECT_NAME,
            MaxComputeConfig.ACCESS_ID,
            MaxComputeConfig.ACCESS_KEY,
        )
    )


class ODPSReader(object):
    def __init__(
        self,
        project,
        access_id,
        access_key,
        endpoint,
        table,
        partition=None,
        num_processes=None,
        options=None,
        transform_fn=None,
        columns=None,
    ):
        """
        Constructs a `ODPSReader` instance.

        Args:
            project: Name of the ODPS project.
            access_id: ODPS user access ID.
            access_key: ODPS user access key.
            endpoint: ODPS cluster endpoint.
            table: ODPS table name.
            partition: ODPS table's partition.
            options: Other options passed to ODPS context.
            num_processes: Number of parallel processes on this worker.
                If `None`, use the number of cores.
            transform_fn: Customized transfrom function
            columns: list of table column names
        """
        super(ODPSReader, self).__init__()

        if table.find(".") > 0:
            project, table = table.split(".")
        if options is None:
            options = {}
        self._project = project
        self._access_id = access_id
        self._access_key = access_key
        self._endpoint = endpoint
        self._table = table
        self._partition = partition
        self._num_processes = num_processes
        _configure_odps_options(self._endpoint, options)
        self._odps_table = ODPS(
            self._access_id, self._access_key, self._project, self._endpoint
        ).get_table(self._table)

        self._transform_fn = transform_fn
        self._columns = columns

    def reset(self, shards):
        self._result_queue = Queue()
        self._index_queues = []
        self._workers = []

        self._shards = []
        self._shard_idx = 0
        self._worker_idx = 0

        for i in range(self._num_processes):
            index_queue = Queue()
            self._index_queues.append(index_queue)

            p = Process(target=self._worker_loop, args=(i,))
            p.daemon = True
            p.start()
            self._workers.append(p)

        self._create_shards(shards)
        for i in range(2 * self._num_processes):
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
        for w in self._workers:
            w.join()
        for q in self._index_queues:
            q.cancel_join_thread()
            q.close()

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
                    if self._transform_fn:
                        record = self._transform_fn(record)
                    records.append(record)
            self._result_queue.put(records)

    def _create_shards(self, shards, shard_size):
        start = shards[0]
        count = shards[1]
        m = count // shard_size
        n = count % shard_size

        for i in range(m):
            self._shards.append((start + i * shard_size, shard_size))
        if n != 0:
            self._shards.append((start + m * shard_size, n))

    def _next_worker_id(self):
        cur_id = self._worker_idx
        self._worker_idx += 1
        if self._worker_idx == self._num_processes:
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

    def record_generator_with_retry(
        self, start, end, columns=None, max_retries=3
    ):
        """Wrap record_generator with retry to avoid ODPS table read failure
        due to network instability.
        """
        retry_count = 0
        while retry_count < max_retries:
            try:
                for record in self.record_generator(start, end, columns):
                    yield record
                break
            except Exception as e:
                if retry_count >= max_retries:
                    raise Exception("Exceeded maximum number of retries")
                logger.warning(
                    "ODPS read exception {} for {} in {}."
                    "Retrying time: {}".format(
                        e, columns, self._table, retry_count
                    )
                )
                time.sleep(5)
                retry_count += 1

    def record_generator(self, start, end, columns=None):
        """Generate records from an ODPS table
        """
        if columns is None:
            columns = self._odps_table.schema.names
        with self._odps_table.open_reader(
            partition=self._partition, reopen=False
        ) as reader:
            for record in reader.read(
                start=start, count=end - start, columns=columns
            ):
                yield [str(record[column]) for column in columns]

    def get_table_size(self):
        with self._odps_table.open_reader(partition=self._partition) as reader:
            return reader.count


class ODPSWriter(object):
    def __init__(
        self,
        project,
        access_id,
        access_key,
        endpoint,
        table,
        columns=None,
        column_types=None,
        options=None,
    ):
        """
        Constructs a `ODPSWriter` instance.

        Args:
            project: Name of the ODPS project.
            access_id: ODPS user access ID.
            access_key: ODPS user access key.
            endpoint: ODPS cluster endpoint.
            table: ODPS table name.
            columns: The list of column names in the table,
                which will be inferred if the table exits.
            column_types" The list of column types in the table,
                which will be inferred if the table exits.
            options: Other options passed to ODPS context.
        """
        super(ODPSWriter, self).__init__()

        if table.find(".") > 0:
            project, table = table.split(".")
        if options is None:
            options = {}
        self._project = project
        self._access_id = access_id
        self._access_key = access_key
        self._endpoint = endpoint
        self._table = table
        self._columns = columns
        self._column_types = column_types
        self._odps_table = None
        _configure_odps_options(self._endpoint, options)
        self._odps_client = ODPS(
            self._access_id, self._access_key, self._project, self._endpoint
        )

    def _initialize_table(self):
        if self._odps_client.exist_table(self._table, self._project):
            self._odps_table = self._odps_client.get_table(
                self._table, self._project
            )
        else:
            if self._columns is None or self._column_types is None:
                raise ValueError(
                    "columns and column_types need to be "
                    "specified for non-existing table."
                )
            schema = Schema.from_lists(
                self._columns, self._column_types, ["worker"], ["string"]
            )
            self._odps_table = self._odps_client.create_table(
                self._table, schema
            )

    def from_iterator(self, records_iter, worker_index):
        if self._odps_table is None:
            self._initialize_table()
        with self._odps_table.open_writer(
            partition="worker=" + str(worker_index), create_partition=True
        ) as writer:
            for records in records_iter:
                writer.write(records)
