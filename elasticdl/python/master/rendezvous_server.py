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
import copy
import time
from threading import Lock

try:
    from horovod.runner.common.util.hosts import (
        get_host_assignments,
        parse_hosts,
    )
    from horovod.runner.http.http_server import RendezvousServer

    _HOROVOD_INSTALLED = True
except ImportError:
    _HOROVOD_INSTALLED = False

_WORKER_SLOT_NUMBER = 1
_HOST_SEP = ","


class HorovodRendezvousServer(object):
    """The rendezvous server can collect worker hosts (ip) to
    help these workers to build an AllReduce ring using `hvd.init`.

    The state transition diagram of the server is:

                    |------------------|
                    |       start      |
                    |next_hosts = None |
                    |------------------|
                            | worker-0 sends the start
                            | message
                            |
                     |------------------|
                 |-- |next_hosts = [0]  |------------------|
                 |   |------------------|                  |
worker-1 sends   |                        worker-0 queries |
the start message|                                  a rank |
    |---------------------| worker-0 queries   |--------------------|
    |next_hosts = [0, 1]  |     a rank         |cur_hosts=next_hosts|
    |                     | ---------------->  |next_hosts=None     |
    |---------------------|                    | ready_hosts adds    |
                                               | the worker         |<---|
                                  |<---------  |--------------------|    |
                worker-2 sends    |                                      |
                the start message |              worker-2 quries         |
                    |-------------------------|  a rank and              |
                    |next_hosts=cur_hosts+[2] |  ready_hosts=cur_hosts   |
                    | ------------------------|  ----------------------->|
    """

    def __init__(self, server_host):
        self._rendezvous_host = server_host
        self._init_attributes()

    def _init_attributes(self):
        self._rendezvous_id = 0
        self._cur_rendezvous_hosts = []
        self._rendezvous_server = RendezvousServer(verbose=True)
        self._rendezvous_port = None
        self._next_rendezvous_hosts = None
        self._ready_worker_hosts = set()
        self._cur_rendezvous_completed = True
        self._lock = Lock()

    def start(self):
        self._rendezvous_port = self._rendezvous_server.start()

    def _init_rendezvous_server(self):
        self._cur_rendezvous_hosts = self._next_rendezvous_hosts
        self._next_rendezvous_hosts = None
        host_alloc_plan = self._get_host_plan()
        self._rendezvous_server.init(host_alloc_plan)
        self._rendezvous_id += 1
        self._cur_rendezvous_completed = False

    def _get_host_plan(self):
        hosts = []
        for host in self._cur_rendezvous_hosts:
            hosts.append(host + ":" + str(_WORKER_SLOT_NUMBER))

        host_infos = parse_hosts(_HOST_SEP.join(hosts))
        host_alloc_plan = get_host_assignments(host_infos, len(host_infos))
        return host_alloc_plan

    def get_rendezvous_host(self):
        return self._rendezvous_host

    def get_rendezvous_port(self):
        return self._rendezvous_port

    def get_worker_host_rank(self, host):
        with self._lock:
            if self._next_rendezvous_hosts and self._cur_rendezvous_completed:
                time.sleep(2)  # Wait 2s for workers to complete rendezvous.
                self._init_rendezvous_server()

            # -1 if host not in worker_hosts list.
            if host not in self._cur_rendezvous_hosts:
                return -1

            if not self._cur_rendezvous_completed:
                self._ready_worker_hosts.add(host)
                # If all active workers in the rendezvous are ready,
                # the server can start to set hosts for the next rendezvous
                if self._ready_worker_hosts == set(self._cur_rendezvous_hosts):
                    self._cur_rendezvous_completed = True
                    self._ready_worker_hosts = set()

            return self._cur_rendezvous_hosts.index(host)

    def get_size(self):
        return len(self._cur_rendezvous_hosts)

    def get_rendezvous_id(self):
        return self._rendezvous_id

    def add_worker(self, worker_host):
        with self._lock:
            if worker_host and worker_host not in self._cur_rendezvous_hosts:
                if self._next_rendezvous_hosts is None:
                    self._next_rendezvous_hosts = copy.deepcopy(
                        self._cur_rendezvous_hosts
                    )
                self._next_rendezvous_hosts.append(worker_host)

    def remove_worker(self, worker_host):
        with self._lock:
            if worker_host in self._cur_rendezvous_hosts:
                if self._next_rendezvous_hosts is None:
                    self._next_rendezvous_hosts = copy.deepcopy(
                        self._cur_rendezvous_hosts
                    )
                self._next_rendezvous_hosts.pop(
                    self._next_rendezvous_hosts.index(worker_host)
                )
