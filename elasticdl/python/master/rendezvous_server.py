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
    def __init__(self, server_host):
        self._rendezvous_host = server_host
        self._rendezvous_id = 0
        self._worker_hosts = []
        self._rendezvous_server = RendezvousServer(verbose=True)
        self._rendezvous_port = None
        self._next_worker_hosts = None
        self._ready_worker_hosts = set()
        self._rendezvous_completed = True
        self._lock = Lock()

    def start(self):
        self._rendezvous_port = self._rendezvous_server.start()

    def set_worker_hosts(self, worker_hosts):
        """
        Set worker hosts into RendezvousServer.

        Args:
            worker_hosts: List of host string.
        """

        if sorted(worker_hosts) != sorted(self._worker_hosts):
            self._next_worker_hosts = worker_hosts

    def _init_rendezvous_server(self):
        self._worker_hosts = self._next_worker_hosts
        self._next_worker_hosts = None
        host_alloc_plan = self._get_host_plan()
        self._rendezvous_server.init(host_alloc_plan)
        self._rendezvous_id += 1
        self._rendezvous_completed = False

    def _get_host_plan(self):
        hosts = []
        for host in self._worker_hosts:
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
            if self._next_worker_hosts and self._rendezvous_completed:
                time.sleep(2)  # Wait 2s for workers to complete rendezvous.
                self._init_rendezvous_server()

            # -1 if host not in worker_hosts list.
            if host not in self._worker_hosts:
                return -1

            if not self._rendezvous_completed:
                self._ready_worker_hosts.add(host)
                # If all active workers in the rendezvous are ready,
                # the server can start to set hosts for the next rendezvous
                if self._ready_worker_hosts == set(self._worker_hosts):
                    self._rendezvous_completed = True
                    self._ready_worker_hosts = set()

            return self._worker_hosts.index(host)

    def get_size(self):
        return len(self._worker_hosts)

    def get_rendezvous_id(self):
        return self._rendezvous_id
