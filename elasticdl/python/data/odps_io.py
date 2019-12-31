import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor as Executor
from queue import Queue

import numpy as np
import odps
from odps import ODPS
from odps.models import Schema

from elasticdl.python.common.constants import ODPSConfig
from elasticdl.python.common.log_utils import default_logger as logger


MAX_DOWNLOAD_SIZE = 20 * 1024 * 1024  # 20MB


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
            ODPSConfig.PROJECT_NAME,
            ODPSConfig.ACCESS_ID,
            ODPSConfig.ACCESS_KEY,
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
        _configure_odps_options(self._endpoint, options)
        self._odps_table = ODPS(
            self._access_id, self._access_key, self._project, self._endpoint
        ).get_table(self._table)

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

        table_size = self.get_table_size()
        if 0 < limit < table_size:
            table_size = limit
        if columns is None:
            columns = self._odps_table.schema.names

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
                    self.read_batch, range_start, range_end, columns
                )
                futures.put(future)

            worker_items_index = self._num_processes

            while not futures.empty():
                if worker_items_index < len(worker_items_with_epoch):
                    range_start = worker_items_with_epoch[worker_items_index]
                    range_end = min(range_start + large_batch_size, table_size)
                    future = executor.submit(
                        self.read_batch, range_start, range_end, columns
                    )
                    futures.put(future)
                    worker_items_index = worker_items_index + 1

                head_future = futures.get()
                records = head_future.result()
                for i in range(0, len(records), batch_size):
                    yield records[i : i + batch_size]  # noqa: E203

    def read_batch(self, start, end, columns=None, max_retries=3):
        """
        Read ODPS table in chosen row range [ `start`, `end` ) with the
        specified columns `columns`.

        Args:
            start: The row index to start reading.
            end: The row index to end reading.
            columns: The list of column to read.
            max_retries : The maximum number of retries in case of exceptions.

        Returns:
            Two-dimension python list with shape: (end - start, len(columns))
        """
        retry_count = 0
        if columns is None:
            columns = self._odps_table.schema.names
        while retry_count < max_retries:
            try:
                batch_record = []
                with self._odps_table.open_reader(
                    partition=self._partition, reopen=True
                ) as reader:
                    for record in reader.read(
                        start=start, count=end - start, columns=columns
                    ):
                        batch_record.append(
                            [str(record[column]) for column in columns]
                        )
                return batch_record
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

    def get_table_size(self):
        with self._odps_table.open_reader(partition=self._partition) as reader:
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

        if table_size < sample_size:
            return 1

        record_size = self._estimate_record_size(columns)
        size_per_batch = record_size * batch_size

        # `size_per_batch * cache_batch_count` will
        # not exceed upper bound but will always greater than 0
        cache_batch_count_estimate = max(
            int(MAX_DOWNLOAD_SIZE / size_per_batch), 1
        )

        return min(cache_batch_count_estimate, max_cache_batch_count)

    def estimate_record_count_each_download(self, columns):
        """Estimate the count of records with 20MB to download
        """
        record_size = self._estimate_record_size(columns)
        max_count = int(MAX_DOWNLOAD_SIZE / record_size)
        return max_count

    def _estimate_record_size(self, columns):
        """Estimate the average size of each record in the ODPS table.
        """
        sample_count = 10
        columns = self._odps_table.schema.names if columns is None else columns
        sample_records = self.read_batch(
            start=0, end=sample_count, columns=columns
        )
        sample_size = _nested_list_size(sample_records)
        record_size = sample_size / sample_count

        return record_size


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
