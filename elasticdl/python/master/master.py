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

from elasticdl.python.master.elasticdl_job_service import ElasticdlJobService
from elasticdl.python.master.pod_manager import PodManager
from elasticdl.python.master.task_manager import TaskManager
from elasticdl_client.common.constants import DistributionStrategy


class Master(object):
    def __init__(self, args):
        self.create_pod_manager_if_needed(args)
        self.create_task_manager_if_needed(args)
        self.create_rendezvous_server_if_needed(args)
        self.create_elasticdl_job_service_if_needed(args)

    def prepare(self):
        if self.pod_manager:
            self.pod_manager.start()
        if self.task_manager:
            self.task_manager.start()
        if self.rendezvous_server:
            self.rendezvous_server.start()
        if self.elasticdl_job_service:
            self.elasticdl_job_service.start()

    def run(self):
        # TODO: Implement run loop here after we have implemented
        #       pod manager or task manager
        if self.elasticdl_job_service:
            return self.elasticdl_job_service.run()
        else:
            return -1

    def create_pod_manager_if_needed(self, args):
        # TODO: set None if args.need_pod_manager is False.
        self.pod_manager = PodManager(args)

    def create_task_manager_if_needed(self, args):
        if args.need_task_manager:
            self.task_manager = TaskManager(args)
        else:
            self.task_manager = None

    def create_rendezvous_server_if_needed(self, args):
        if args.distribution_strategy != DistributionStrategy.ALLREDUCE:
            self.rendezvous_server = None
        # TODO: create HorovodRendezvousServer
        self.rendezvous_server = None

    def create_elasticdl_job_service_if_needed(self, args):
        if args.need_elasticdl_job_service:
            self.elasticdl_job_service = ElasticdlJobService(
                args, self.task_manager
            )
        else:
            self.elasticdl_job_service = None
