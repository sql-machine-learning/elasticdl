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

from elasticdl.python.common.constants import InstanceManagerStatus
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.master.elasticdl_job_service import ElasticdlJobService
from elasticdl.python.master.pod_manager import create_pod_manager
from elasticdl.python.master.servicer import create_master_service
from elasticdl.python.master.task_manager import TaskManager
from elasticdl_client.common.constants import DistributionStrategy


class Master(object):
    def __init__(self, args):
        self.create_task_manager_if_needed(args)
        self.create_rendezvous_server_if_needed(args)
        self.create_pod_manager_if_needed(args)
        self.create_elasticdl_job_service_if_needed(args)
        self.create_master_grpc_service(args)
        self._args = args
        self._exit_code = 0

    def prepare(self):
        if self.task_manager:
            self.task_manager.start()
        if self.rendezvous_server:
            self.rendezvous_server.start()
        if self.pod_manager:
            if self.elasticdl_job_service:
                command = self.elasticdl_job_service.get_ps_worker_command()
                self.pod_manager.set_up(
                    worker_command=command,
                    worker_args=self.elasticdl_job_service.get_worker_args(
                        self._args
                    ),
                    ps_command=command,
                    ps_args=self.elasticdl_job_service.get_ps_args(self._args),
                )
            else:
                # TODO: Get the Pod arguments from the input
                # args directly
                pass
            self.pod_manager.start()
        if self.elasticdl_job_service:
            self.elasticdl_job_service.start()

        # Start the master GRPC server
        logger.info("Starting master RPC server")
        self._master_server.start()
        logger.info("Master RPC server started")

    def run(self):
        """
        The main loop of master.
        Dispatch the tasks to the workers until all the tasks are completed.
        """
        try:
            while True:
                if self.task_manager.finished():
                    if self.pod_manager:
                        self.pod_manager.update_status(
                            InstanceManagerStatus.FINISHED
                        )
                    break
                if self.pod_manager.all_workers_exited:
                    raise Exception(
                        "All workers exited but there also are",
                        "unfinished tasks",
                    )
                time.sleep(30)
        except KeyboardInterrupt:
            self.logger.warning("Server stopping")
        finally:
            self.stop()
        return self._exit_code

    def stop(self):
        """
        Stop all the components.
        Make sure that the created services and components are shut down.
        """
        logger.info("Stopping master")
        logger.info("Stopping RPC server")
        self._master_server.stop(None)  # grace = None
        logger.info("RPC server stopped")
        logger.info("Master stopped")

    def create_pod_manager_if_needed(self, args):
        # TODO: set None if args.need_pod_manager is False.
        self.pod_manager = create_pod_manager(
            args, self.task_manager, self.rendezvous_server
        )

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
                args=args,
                task_manager=self.task_manager,
                pod_manager=self.pod_manager,
            )
        else:
            self.elasticdl_job_service = None

    def create_master_grpc_service(self, args):
        # TODO: Move the rendezvous_server out of elasticdl_job_service
        rendezvous_server = (
            self.elasticdl_job_service.rendezvous_server
            if self.elasticdl_job_service
            else None
        )
        evaluation_service = (
            self.elasticdl_job_service.evaluation_service
            if self.elasticdl_job_service
            else None
        )

        self._master_server = create_master_service(
            args.port,
            self.task_manager,
            self.pod_manager,
            rendezvous_server,
            evaluation_service,
        )
