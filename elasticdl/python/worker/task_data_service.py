# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import time

import tensorflow as tf

from elasticai_api.proto import elasticai_api_pb2
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.data.reader.data_reader_factory import create_data_reader


class TaskDataService(object):
    def __init__(
        self,
        data_shard_service,
        custom_data_reader=None,
        data_reader_params=None,
        data_origin=None,
    ):
        self._data_shard_service = data_shard_service
        self._create_data_reader_fn = create_data_reader
        if custom_data_reader is not None:
            self._create_data_reader_fn = custom_data_reader
        self._lock = threading.Lock()
        self.train_end_callback_task = None
        if data_reader_params:
            self.data_reader = self._create_data_reader_fn(
                data_origin=data_origin, **data_reader_params
            )
        else:
            self.data_reader = self._create_data_reader_fn(
                data_origin=data_origin
            )
        self.current_eval_task = None

    def get_current_task(self):
        return self._data_shard_service.get_current_task()

    def report_record_done(self, count, err_msg=""):
        self._data_shard_service.report_batch_done(count, err_msg)

    def get_dataset_gen(self, task):
        """
        If a task exists, this creates a generator, which could be used to
        creating a `tf.data.Dataset` object in further.
        """
        if not task:
            return None
        tasks = [task]

        def gen():
            for task in tasks:
                for data in self.data_reader.read_records(task):
                    if data:
                        yield data

        return gen

    def get_dataset_by_task(self, task):
        if task is None:
            return None
        gen = self.get_dataset_gen(task)
        dataset = tf.data.Dataset.from_generator(
            gen, self.data_reader.records_output_types
        )
        return dataset

    def get_train_end_callback_task(self):
        while True:
            task = self._data_shard_service.get_task()
            if task.type == elasticai_api_pb2.TRAIN_END_CALLBACK:
                self.train_end_callback_task = task
                return task
            elif task.type == elasticai_api_pb2.WAIT:
                # The worker can only do the callback task until
                # the training loop finishes.
                logger.info("Waiting more tasks")
                time.sleep(5)
            else:
                return None

    def get_dataset(self):
        """
        If there's more data, this creates a `tf.data.Dataset` object.
        Otherwise, this returns `None`.
        """
        ds = tf.data.Dataset.from_generator(
            self._gen, self.data_reader.records_output_types
        )
        return ds

    def _gen(self):
        """
        A generator supports the iter() protocol (e.g. a generator function),
        used to create a `tf.data.Dataset` object from a list of tasks.
        """
        while True:
            task = self._data_shard_service.get_task()
            if task.type != elasticai_api_pb2.TRAINING:
                break

            for data in self.data_reader.read_records(task):
                if data:
                    yield data

    def get_eval_dataset(self):
        def _gen():
            task = self._data_shard_service.get_task(
                elasticai_api_pb2.EVALUATION
            )
            if task.type != elasticai_api_pb2.EVALUATION:
                return
            logger.info("the evaluation task_id: %d" % task.task_id)
            self.current_eval_task = task
            for data in self.data_reader.read_records(task):
                if data:
                    yield data

        dataset = tf.data.Dataset.from_generator(
            _gen, self.data_reader.records_output_types
        )
        return dataset
