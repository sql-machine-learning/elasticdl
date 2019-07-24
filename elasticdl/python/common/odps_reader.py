import random
from concurrent.futures import ThreadPoolExecutor as Executor
from queue import Queue

import numpy as np
import odps
from odps import ODPS


def _nested_list_size(l):
    """
    Obtains the memory size for the nested list.
    """
    import sys

    total = sys.getsizeof(l)
    for i in l:
        if isinstance(i, list):
            total += _nested_list_size(i)
        else:
            total += sys.getsizeof(i)

    return total


def _read_odps_one_shot(
    project,
    access_id,
    access_key,
    endpoint,
    table,
    partition,
    start,
    end,
    columns,
    max_retries=3,
):
    """
    Read ODPS table in chosen row range [ `start`, `end` ) with the specified
    columns `columns`.

    Args:
        project: The ODPS project.
        access_id: The ODPS user access ID.
        access_key: The ODPS user access key.
        endpoint: The ODPS cluster endpoint.
        table: The ODPS table name.
        partition: The ODPS table's partition. Default is `None` if the
            table is not partitioned.
        start: The row index to start reading.
        end: The row index to end reading.
        columns: The list of column to read.
        max_retries : The maximum number of retries in case of exceptions.

    Returns: Two-dimension python list with shape: (end - start, len(columns))
    """
    odps_table = ODPS(access_id, access_key, project, endpoint).get_table(
        table
    )

    retry_count = 0

    while retry_count < max_retries:
        try:
            batch_record = []
            with odps_table.open_reader(
                partition=partition, reopen=True
            ) as reader:
                for record in reader.read(
                    start=start, count=end - start, columns=columns
                ):
                    batch_record.append([record[column] for column in columns])

            return batch_record

        except Exception as e:
            import time

            if retry_count >= max_retries:
                raise
            print(
                "ODPS read exception {} for {} in {}. retrying {} time".format(
                    e, columns, table, retry_count
                )
            )
            time.sleep(5)
            retry_count += 1


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
        self._options = options

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
            and "service.odps.aliyun-inc.com/api" in self._endpoint
        ):
            odps.options.tunnel.endpoint = "http://dt.odps.aliyun-inc.com"

    def to_iterator(
        self,
        num_workers,
        worker_index,
        batch_size,
        epochs=1,
        shuffle=False,
        columns=None,
        cache_batch_count=None,
        limit=-1,
    ):
        """
        Load slices of ODPS table (partition of table if `partition`
        was specified) data with Python Generator.

        Args:
            num_workers: Total number of worker in the cluster.
            worker_index: Current index of the worker in the cluster.
            batch_size: Size of a slice.
            epochs: Repeat the data for this many times.
            shuffle: Whether to shuffle the data or rows.
            columns: The list of columns to load. If `None`,
                use all schema names of ODPS table.
            cache_batch_count: The cache batch count.
            limit: The limit for the table size to load.
        """
        if not worker_index < num_workers:
            raise ValueError(
                "index of worker should be less than number of worker"
            )
        if not batch_size > 0:
            raise ValueError("batch_size should be positive")
        odps_table = ODPS(
            self._access_id, self._access_key, self._project, self._endpoint
        ).get_table(self._table)
        table_size = self._count_table_size(odps_table)
        if 0 < limit < table_size:
            table_size = limit
        if columns is None:
            columns = odps_table.schema.names

        if cache_batch_count is None:
            cache_batch_count = self._estimate_cache_batch_count(
                columns=columns, table_size=table_size, batch_size=batch_size
            )

        large_batch_size = batch_size * cache_batch_count

        overall_items = range(0, table_size, large_batch_size)

        if len(overall_items) < num_workers:
            overall_items = range(0, table_size, int(table_size / num_workers))

        worker_items = list(
            np.array_split(np.asarray(overall_items), num_workers)[
                worker_index
            ]
        )
        if shuffle:
            random.shuffle(worker_items)
        worker_items_with_epoch = worker_items * epochs

        # `worker_items_with_epoch` is the total number of batches
        # that needs to be read and the worker number should not
        # be larger than `worker_items_with_epoch`
        if self._num_processes is None:
            self._num_processes = min(8, len(worker_items_with_epoch))
        else:
            self._num_processes = min(
                self._num_processes, len(worker_items_with_epoch)
            )

        if self._num_processes == 0:
            raise ValueError(
                "Total worker number is 0. Please check if table has data."
            )

        with Executor(max_workers=self._num_processes) as executor:

            futures = Queue()
            # Initialize concurrently running processes according
            # to `num_processes`
            for i in range(self._num_processes):
                range_start = worker_items_with_epoch[i]
                range_end = min(range_start + large_batch_size, table_size)
                future = executor.submit(
                    _read_odps_one_shot,
                    self._project,
                    self._access_id,
                    self._access_key,
                    self._endpoint,
                    self._table,
                    self._partition,
                    range_start,
                    range_end,
                    columns,
                )
                futures.put(future)

            worker_items_index = self._num_processes

            while not futures.empty():
                if worker_items_index < len(worker_items_with_epoch):
                    range_start = worker_items_with_epoch[worker_items_index]
                    range_end = min(range_start + large_batch_size, table_size)
                    future = executor.submit(
                        _read_odps_one_shot,
                        self._project,
                        self._access_id,
                        self._access_key,
                        self._endpoint,
                        self._table,
                        self._partition,
                        range_start,
                        range_end,
                        columns,
                    )
                    futures.put(future)
                    worker_items_index = worker_items_index + 1

                head_future = futures.get()
                records = head_future.result()
                for i in range(0, len(records), batch_size):
                    yield records[i : i + batch_size]  # noqa: E203

    def _count_table_size(self, odps_table):
        with odps_table.open_reader(partition=self._partition) as reader:
            return reader.count

    def _estimate_cache_batch_count(self, columns, table_size, batch_size):
        """
        This function calculates the appropriate cache batch size
        when we download from ODPS, if batch size is small, we will
        repeatedly create http connection and download small chunk of
        data. To read more efficiently, we will read
        `batch_size * cache_batch_count` lines of data.
        However, determining a proper `cache_batch_count` is non-trivial.
        Our heuristic now is to set a per download upper bound.
        """

        sample_size = 10
        max_cache_batch_count = 50
        upper_bound = 20 * 1000000

        if table_size < sample_size:
            return 1

        batch = _read_odps_one_shot(
            project=self._project,
            access_id=self._access_id,
            access_key=self._access_key,
            endpoint=self._endpoint,
            table=self._table,
            partition=self._partition,
            start=0,
            end=sample_size,
            columns=columns,
        )

        size_sample = _nested_list_size(batch)
        size_per_batch = size_sample * batch_size / sample_size

        # `size_per_batch * cache_batch_count` will
        # not exceed upper bound but will always greater than 0
        cache_batch_count_estimate = max(int(upper_bound / size_per_batch), 1)

        return min(cache_batch_count_estimate, max_cache_batch_count)
