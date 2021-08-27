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

from elasticdl.python.common.k8s_client import PodType
from elasticdl.python.common.log_utils import default_logger as logger

_WAIT_SYNC_TINEOUT = 3600


class WorkerSyncObjects(object):
    def __init__(self, pod_manager):
        self._pod_manager = pod_manager
        self._sync_objs = {}
        self._wait_sync_objs = []
        self._lock = threading.Lock()
        self._sync_start_time = {}
        self.timeout = _WAIT_SYNC_TINEOUT
        threading.Thread(
            target=self.delete_sync_timeout_worker,
            name="sync timeout worker monitor",
            daemon=True,
        ).start()

    def worker_sync(self, sync_name, worker_id):
        with self._lock:
            if sync_name not in self._sync_objs:
                self._sync_objs[
                    sync_name
                ] = self._pod_manager.get_running_pod_ids(PodType.WORKER)
                logger.info(
                    "New worker sync {} added for worker {}".format(
                        sync_name, self._sync_objs[sync_name]
                    )
                )
                self._sync_start_time[sync_name] = time.time()
            if worker_id in self._sync_objs[sync_name]:
                self._sync_objs[sync_name].remove(worker_id)
                logger.info(
                    "{}: worker {} synced. Remaining {}".format(
                        sync_name, worker_id, self._sync_objs[sync_name]
                    )
                )
                if len(self._sync_objs[sync_name]) == 0:
                    self._sync_start_time.pop(sync_name)
                    logger.info("Worker sync {} done.".format(sync_name))
            if len(self._sync_objs[sync_name]) == 0:
                return True
        return False

    def wait_worker_sync(self, sync_name, notify):
        with self._lock:
            if sync_name in self._wait_sync_objs:
                return True
            if notify:
                self._wait_sync_objs.append(sync_name)
                logger.info("Worker sync {} notified".format(sync_name))
                return True
            else:
                return False

    def non_running_worker_update(self, worker_id):
        with self._lock:
            for sync_name in self._sync_objs:
                if worker_id in self._sync_objs[sync_name]:
                    self._sync_objs[sync_name].remove(worker_id)
                    logger.info(
                        "Worker {} not running, removed from {}. "
                        "Remaining {}".format(
                            worker_id, sync_name, self._sync_objs[sync_name]
                        )
                    )
                    if len(self._sync_objs[sync_name]) == 0:
                        logger.info("Worker sync {} done.".format(sync_name))

    def delete_worker_sync(self, sync_name, delete_all=False):
        with self._lock:
            if not delete_all:
                if sync_name in self._sync_objs:
                    del self._sync_objs[sync_name]
                    logger.info("Worker sync {} deleted".format(sync_name))
                if sync_name in self._wait_sync_objs:
                    self._wait_sync_objs.remove(sync_name)
                    logger.info(
                        "Worker wait sync {} deleted".format(sync_name)
                    )
            if delete_all:
                self._sync_objs = {}
                self._wait_sync_objs = []
                logger.info("All worker syncs deleted.")

    def delete_sync_timeout_worker(self):
        while True:
            timeout_syncs = []
            timeout_workers = set()
            with self._lock:
                for sync_name, start_time in self._sync_start_time.items():
                    if time.time() - start_time > self.timeout:
                        timeout_syncs.append(sync_name)
                        for worker_id in self._sync_objs[sync_name]:
                            timeout_workers.add(worker_id)
                for sync_name in timeout_syncs:
                    self._sync_start_time.pop(sync_name)

            for worker_id in timeout_workers:
                logger.info("Remove timeout worker {}".format(worker_id))
                self._pod_manager.remove_worker(worker_id)
            time.sleep(15)
