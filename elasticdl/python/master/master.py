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

import os
import time

from elasticdl.python.common.constants import PodManagerStatus
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.master.elasticdl_job_service import ElasticdlJobService
from elasticdl.python.master.pod_event_callbacks import (
    RendezvousServiceRefreshCallback,
    TaskRescheduleCallback,
)
from elasticdl.python.master.pod_manager import create_pod_manager
from elasticdl.python.master.rendezvous_server import HorovodRendezvousServer
from elasticdl.python.master.servicer import create_master_service
from elasticdl.python.master.task_manager import TaskManager
from elasticdl_client.common.constants import DistributionStrategy


class Master(object):
    def __init__(self, args):
        self.create_pod_manager_if_needed(args)
        self.create_task_manager_if_needed(args)
        self.create_rendezvous_server_if_needed(args)
        self.create_elasticdl_job_service_if_needed(args)
        self.create_master_grpc_service(args)
        self._args = args
        self._exit_code = 0

    def prepare(self):
        self.validate()
        # Composite the components
        if self.task_manager and self.pod_manager:
            self.task_manager.set_task_timeout_callback(
                self.pod_manager._remove_worker
            )
        if self.pod_manager:
            self._set_command_in_pod_manager()
            # Add PodEventCallbacks for the listeners of Pod events.
            if self.task_manager:
                self.pod_manager.add_pod_event_callback(
                    TaskRescheduleCallback(self.task_manager)
                )
            if self.rendezvous_server:
                self.pod_manager.add_pod_event_callback(
                    RendezvousServiceRefreshCallback(self.rendezvous_server)
                )

        # Start the components one by one
        if self.task_manager:
            self.task_manager.start()
        if self.rendezvous_server:
            self.rendezvous_server.start()
        if self.pod_manager:
            self.pod_manager.start()
        if self.elasticdl_job_service:
            self.elasticdl_job_service.start()

        # Start the master GRPC server
        logger.info("Starting master RPC server")
        self._master_server.start()
        logger.info("Master RPC server started")

    def _set_command_in_pod_manager(self):
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
        elif self._args.job_command:
            self.pod_manager.set_up(
                worker_command=["/bin/bash"],
                worker_args=["-c", self._args.job_command],
                ps_command=["/bin/bash"],
                ps_args=["-c", self._args.job_command],
            )
        else:
            raise ValueError(
                "job_command is necessary if there is no elasticdl job "
                "service."
            )

    def run(self):
        """
        The main loop of master.
        Dispatch the tasks to the workers until all the tasks are completed.
        """
        try:
            while True:
                if self.pod_manager and self.pod_manager.all_workers_exited:
                    if self.task_manager and not self.task_manager.finished():
                        logger.warning(
                            "All workers exited but there also are "
                            "unfinished tasks",
                        )
                    if self.pod_manager.all_workers_failed:
                        raise RuntimeError("All workers failed")
                    self.pod_manager.update_status(PodManagerStatus.FINISHED)
                    break
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
        if args.need_pod_manager:
            self.pod_manager = create_pod_manager(args)
        else:
            self.pod_manager = None

    def create_task_manager_if_needed(self, args):
        if args.need_task_manager:
            self.task_manager = TaskManager(args)
        else:
            self.task_manager = None

    def create_rendezvous_server_if_needed(self, args):
        if args.distribution_strategy != DistributionStrategy.ALLREDUCE:
            self.rendezvous_server = None
        else:
            master_ip = os.getenv("MY_POD_IP", "localhost")
            self.rendezvous_server = HorovodRendezvousServer(master_ip)

    def create_elasticdl_job_service_if_needed(self, args):
        if args.need_elasticdl_job_service:
            # TODO: Remove rendezvous server after rafactoring the pod
            # manager.
            self.elasticdl_job_service = ElasticdlJobService(
                args=args,
                task_manager=self.task_manager,
                rendezvous_server=self.rendezvous_server,
            )
        else:
            self.elasticdl_job_service = None

    def create_master_grpc_service(self, args):
        evaluation_service = (
            self.elasticdl_job_service.evaluation_service
            if self.elasticdl_job_service
            else None
        )

        self._master_server = create_master_service(
            args.port,
            self.task_manager,
            self.pod_manager,
            self.rendezvous_server,
            evaluation_service,
        )

    def validate(self):
        """
        Check if the master has a valid configuration.
        If not, raise exception.
        """
        need_pod_manager = (
            (self.task_manager and self.task_manager.support_fault_tolerance)
            or self.rendezvous_server
            or self.elasticdl_job_service
        )
        if need_pod_manager and not self.pod_manager:
            raise Exception("Pod manager is required.")
        if self.elasticdl_job_service and not (
            self.task_manager and self.task_manager.support_fault_tolerance
        ):
            raise Exception(
                "Task manager with fault tolerance is required for ",
                "elasticdl job service.",
            )
