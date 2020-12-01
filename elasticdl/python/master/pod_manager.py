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
import itertools
import os
import threading
import time
from collections import Counter

from kubernetes.client import V1EnvVar

from elasticdl.python.common import k8s_client as k8s
from elasticdl.python.common.constants import (
    PodManagerStatus,
    PodStatus,
    WorkerEnv,
)
from elasticdl.python.common.k8s_client import PodType
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.common.model_utils import get_dict_from_params_str
from elasticdl_client.common.args import parse_envs
from elasticdl_client.common.constants import (
    BashCommandTemplate,
    ClusterSpecConfig,
)

_SERVICE_ADDR_SEP = ","


def _get_addrs(num_addrs, addr_get_fn):
    """
    Get `num_addrs` addresses and then concatenate
    them to a comma separated string.
    """
    addrs = []
    for addr_id in range(num_addrs):
        addrs.append(addr_get_fn(addr_id))
    return _SERVICE_ADDR_SEP.join(addrs)


def _parse_worker_pod_priority(num_workers, worker_pod_priority):
    res = {}
    if isinstance(worker_pod_priority, str) and "high=" in worker_pod_priority:
        try:
            fraction = float(worker_pod_priority.split("=")[1])
            high_count = int(num_workers * fraction)
            for i in range(num_workers):
                if i < high_count:
                    res[i] = "high"
                else:
                    res[i] = "low"
        except Exception:
            logger.warning(
                "Please check the input worker pod priority format,"
                "e.g. high=0.5  The config is no use, and ElasticDL sets "
                "low priority for all worker pods by default."
            )
            for i in range(num_workers):
                res[i] = None
    else:
        for i in range(num_workers):
            res[i] = worker_pod_priority
    return res


def _should_relaunch_killed_pod(evt_obj):
    """
    Check whether to relaunch the failed pod according to the kubernetes event.
    For the killed pods, we will try to relaunch them except the
    OOM ones.
    """
    return (
        evt_obj.status.container_statuses
        and evt_obj.status.container_statuses[0].state.terminated
        and evt_obj.status.container_statuses[0].state.terminated.exit_code
        == 137
        and evt_obj.status.container_statuses[0].state.terminated.reason
        != "OOMKilled"
    )


def get_image_cluster_spec(cluster_spec):
    if cluster_spec:
        filename = os.path.basename(cluster_spec)
        image_cluster_spec = os.path.join(
            ClusterSpecConfig.CLUSTER_SPEC_DIR, filename
        )
        return image_cluster_spec
    return cluster_spec


# TODO: After decouple PodManager with TaskManager and RendezvousServer
# with PodEventCallback, we can remove these two parameter from this
# factory method and the __init__ method of PodManager.
def create_pod_manager(
    args, task_manager, rendezvous_server,
):
    pod_manager = None

    master_ip = os.getenv("MY_POD_IP", "localhost")
    master_addr = "%s:%d" % (master_ip, args.port)
    if args.num_workers:
        assert args.worker_image, "Worker image cannot be empty"

        env_dict = parse_envs(args.envs)
        env = []
        for key in env_dict:
            env.append(V1EnvVar(name=key, value=env_dict[key]))
        env.append(V1EnvVar(name=WorkerEnv.MASTER_ADDR, value=master_addr))

        kwargs = get_dict_from_params_str(args.aux_params)
        disable_relaunch = kwargs.get("disable_relaunch", False)
        cluster_spec = get_image_cluster_spec(args.cluster_spec)

        pod_manager = PodManager(
            task_manager=task_manager,
            rendezvous_server=rendezvous_server,
            job_name=args.job_name,
            image_name=args.worker_image,
            namespace=args.namespace,
            num_workers=args.num_workers,
            worker_resource_request=args.worker_resource_request,
            worker_resource_limit=args.worker_resource_limit,
            worker_pod_priority=args.worker_pod_priority,
            num_ps=args.num_ps_pods,
            ps_resource_request=args.ps_resource_request,
            ps_resource_limit=args.ps_resource_limit,
            ps_pod_priority=args.ps_pod_priority,
            volume=args.volume,
            image_pull_policy=args.image_pull_policy,
            restart_policy=args.restart_policy,
            cluster_spec=cluster_spec,
            cluster_spec_json=args.cluster_spec_json,
            envs=env,
            disable_relaunch=disable_relaunch,
            log_file_path=args.log_file_path,
        )

    return pod_manager


class PodManager(object):
    def __init__(
        self,
        task_manager,
        rendezvous_server=None,
        num_workers=1,
        worker_resource_request="cpu=1,memory=4096Mi",
        worker_resource_limit="cpu=1,memory=4096Mi",
        worker_pod_priority=None,
        num_ps=0,
        ps_resource_request="cpu=1,memory=4096Mi",
        ps_resource_limit="cpu=1,memory=4096Mi",
        ps_pod_priority=None,
        volume=None,
        image_pull_policy=None,
        restart_policy="Never",
        envs=None,
        disable_relaunch=False,
        log_file_path=None,
        **kwargs
    ):
        self._num_workers = num_workers
        self._worker_resource_request = worker_resource_request
        self._worker_resource_limit = worker_resource_limit
        self._worker_pod_priority = _parse_worker_pod_priority(
            self._num_workers, worker_pod_priority
        )

        self._num_ps = num_ps
        self._ps_resource_request = ps_resource_request
        self._ps_resource_limit = ps_resource_limit
        self._ps_pod_priority = ps_pod_priority

        self._restart_policy = restart_policy
        self._volume = volume
        self._image_pull_policy = image_pull_policy
        self._envs = envs
        self._task_manager = task_manager
        self._rendezvous_server = rendezvous_server
        self._next_worker_id_fn = itertools.count().__next__
        self._log_file_path = log_file_path

        # Protects followed variables, which are accessed from event_cb.
        self._lock = threading.Lock()

        self._init_job_pod_status()

        if disable_relaunch:
            self._k8s_client = k8s.Client(**kwargs)
        else:
            self._k8s_client = k8s.Client(
                event_callback=self._event_cb,
                periodic_call_func=self._process_worker,
                **kwargs
            )
        self._ps_addrs = _get_addrs(
            self._num_ps, self._k8s_client.get_ps_service_address
        )
        self._worker_addrs = []
        self._pod_event_callbacks = []
        self._worker_command = None
        self._worker_args = None
        self._ps_command = None
        self._ps_args = None

    def set_up(
        self,
        worker_command=None,
        worker_args=None,
        ps_command=None,
        ps_args=None,
    ):
        self._worker_command = worker_command
        self._worker_args = worker_args
        self._ps_command = ps_command
        self._ps_args = ps_args

    def start(self):
        self._k8s_client.start_watch_events()
        self.update_status(PodManagerStatus.PENDING)
        if self._num_ps > 0:
            logger.info("num ps pods : {}".format(self._num_ps))
            self.start_parameter_servers()
        self.start_workers()
        self.update_status(PodManagerStatus.RUNNING)

    def add_pod_event_callback(self, pod_event_callback):
        self._pod_event_callbacks.append(pod_event_callback)

    def _init_job_pod_status(self):
        # worker id to (pod name, ip, phase) mapping
        # phase: None/Pending/Running/Succeeded/Failed/Unknown
        #   None: worker was just launched, haven't received event yet.
        #   Pending: worker pod not started yet
        #   Running: worker pod is running
        #   Succeeded: worker pod finishes all tasks and terminates with
        #       no issue.
        #   Failed: worker pod is killed for some reason
        #   Unknown: unknown
        self._worker_pods_ip_phase = {}
        # pod name to worker id mapping
        self._worker_pod_name_to_id = {}
        # worker ids whose pods are not created
        self._not_created_worker_id = []

        self._relaunch_deleted_live_worker = True

        self._ps_pods_phase = {}
        self._ps_pod_name_to_id = {}
        self._relaunch_deleted_live_ps = True

        self._failed_pods = []
        self.all_workers_exited = False

    def _process_worker(self):
        need_process = True
        while need_process and self._not_created_worker_id:
            worker_id = self._not_created_worker_id.pop()
            # Try to create a worker pod with id as worker_id
            need_process = self._start_worker(worker_id)

    def _start_worker(self, worker_id):
        logger.info("Starting worker: %d" % worker_id)
        bash_command = self._worker_args[1]
        if self._ps_addrs:
            bash_command += " --ps_addrs {}".format(self._ps_addrs)
        if self._log_file_path:
            bash_command += BashCommandTemplate.REDIRECTION.format(
                self._log_file_path
            )
        for extra_arg in self._worker_args[2:]:
            bash_command += " {}".format(extra_arg)
        worker_args = [self._worker_args[0], bash_command]
        envs = copy.deepcopy(self._envs)
        envs.append(V1EnvVar(name=WorkerEnv.WORKER_ID, value=str(worker_id)))
        with self._lock:
            pod = self._k8s_client.create_worker(
                worker_id=worker_id,
                resource_requests=self._worker_resource_request,
                resource_limits=self._worker_resource_limit,
                pod_priority=self._worker_pod_priority[worker_id],
                termination_period=1,
                volume=self._volume,
                image_pull_policy=self._image_pull_policy,
                command=self._worker_command,
                args=worker_args,
                restart_policy=self._restart_policy,
                ps_addrs=self._ps_addrs,
                envs=envs,
            )
            if pod:
                name = pod.metadata.name
                self._worker_pod_name_to_id[name] = worker_id
                self._worker_pods_ip_phase[worker_id] = (name, None, None)
                return True
            else:
                self._not_created_worker_id.append(worker_id)
                return False

    def _start_ps(self, ps_id):
        logger.info("Starting PS: %d" % ps_id)
        bash_command = self._ps_args[1]
        bash_command += " --ps_id {}".format(ps_id)
        if self._log_file_path:
            bash_command += BashCommandTemplate.REDIRECTION.format(
                self._log_file_path
            )
        ps_args = [self._ps_args[0], bash_command]
        with self._lock:
            while True:
                pod = self._create_ps_pod(ps_id, ps_args)
                if pod:
                    break
                # TODO: should we fail the job when ps pods fail to
                #       create for a long time?
                logger.error(
                    "Creating PS fails and will try again."
                    "ps_id: {}, ps_args: {}.".format(ps_id, ps_args)
                )
                time.sleep(15)
            name = pod.metadata.name
            self._ps_pod_name_to_id[name] = ps_id
            self._ps_pods_phase[ps_id] = (name, None)
            self._k8s_client.create_ps_service(ps_id)

    def _create_ps_pod(self, ps_id, ps_args):
        return self._k8s_client.create_ps(
            ps_id=ps_id,
            resource_requests=self._ps_resource_request,
            resource_limits=self._ps_resource_limit,
            pod_priority=self._ps_pod_priority,
            volume=self._volume,
            image_pull_policy=self._image_pull_policy,
            command=self._ps_command,
            args=ps_args,
            restart_policy=self._restart_policy,
            envs=copy.deepcopy(self._envs),
        )

    def update_status(self, status):
        master_name = self._k8s_client.get_master_pod_name()
        self._k8s_client.patch_labels_to_pod(
            master_name, labels_dict={"status": status}
        )

    def start_workers(self):
        for _ in range(self._num_workers):
            self._start_worker(self._next_worker_id_fn())

    def start_parameter_servers(self):
        for i in range(self._num_ps):
            self._start_ps(i)

    def _remove_worker(self, worker_id):
        logger.info("Removing worker: %d", worker_id)
        with self._lock:
            if worker_id not in self._worker_pods_ip_phase:
                logger.error("Unknown worker id: %s" % worker_id)
                return

        # TODO: change _k8s_client to accept pod name instead of worker id.
        self._k8s_client.delete_worker(worker_id)

    def _remove_parameter_server(self, ps_id):
        logger.info("Removing PS: %d", ps_id)
        with self._lock:
            if ps_id not in self._ps_pods_phase:
                logger.error("Unknown PS id: %s" % ps_id)
                return

        self._k8s_client.delete_ps(ps_id)

    def stop_relaunch_and_remove_pods(self, pod_type):
        ids = []
        if pod_type == PodType.WORKER:
            self._relaunch_deleted_live_worker = False
            ids = self._worker_pods_ip_phase
            delete_func = self._k8s_client.delete_worker
        elif pod_type == PodType.PS:
            self._relaunch_deleted_live_ps = False
            ids = self._ps_pods_phase
            delete_func = self._k8s_client.delete_ps
        with self._lock:
            for id in ids:
                delete_func(id)

    def get_pod_counter(self, pod_type):
        with self._lock:
            worker_counter = Counter(
                [v for _, _, v in self._worker_pods_ip_phase.values()]
            )
            ps_counter = Counter([v for _, v in self._ps_pods_phase.values()])

        if pod_type == PodType.WORKER:
            return worker_counter
        elif pod_type == PodType.PS:
            return ps_counter
        else:
            return None

    # TODO: Refactor the process of _event_cb to publish
    # PodStarted / PodSucceded / PodFailed / PodDeleted
    # events and invoke the corresponding callbacks.
    def _event_cb(self, event):
        evt_obj = event.get("object")
        evt_type = event.get("type")
        if not evt_obj or not evt_type:
            logger.error("Event doesn't have object or type: %s" % event)
            return

        if evt_obj.kind != "Pod":
            # We only care about pod related events
            return

        pod_name = evt_obj.metadata.name
        pod_ip = evt_obj.status.pod_ip
        phase = evt_obj.status.phase
        if pod_name == self._k8s_client.get_master_pod_name():
            # No need to care about master pod
            return

        relaunch_worker = False
        relaunch_ps = False
        worker_id = -1
        ps_id = -1
        with self._lock:
            if pod_name in self._failed_pods:
                return

            # For the failed worker, reassign the its tasks to others.
            # Check whether to relaunch the worker.
            relaunch_failed_pod = False
            if evt_type == "MODIFIED" and phase == "Failed":
                self._failed_pods.append(pod_name)
                worker_id = self._worker_pod_name_to_id.get(pod_name, None)
                if worker_id is not None:
                    # Recover tasks when the worker failed
                    self._task_manager.recover_tasks(worker_id)

                if _should_relaunch_killed_pod(evt_obj):
                    relaunch_failed_pod = True
                    logger.info(
                        "Pod %s is killed with reason %s."
                        % (
                            pod_name,
                            evt_obj.status.container_statuses[
                                0
                            ].state.terminated.reason,
                        )
                    )

            if pod_name in self._worker_pod_name_to_id:
                # If the pod related with the event is a worker
                worker_id = self._worker_pod_name_to_id.get(pod_name)
                self._worker_pods_ip_phase[worker_id] = (
                    pod_name,
                    pod_ip,
                    phase,
                )
                if evt_type == "DELETED" or relaunch_failed_pod:
                    del self._worker_pods_ip_phase[worker_id]
                    del self._worker_pod_name_to_id[pod_name]

                    # If a deleted pod was not "Succeeded", relaunch a worker.
                    relaunch_worker = (
                        self._relaunch_deleted_live_worker
                        and phase != "Succeeded"
                    )
                else:
                    self.check_all_workers_exited()

            elif pod_name in self._ps_pod_name_to_id:
                # If the pod related with the event is a PS
                ps_id = self._ps_pod_name_to_id.get(pod_name)
                self._ps_pods_phase[ps_id] = (pod_name, phase)
                if evt_type == "DELETED" or relaunch_failed_pod:
                    del self._ps_pods_phase[ps_id]
                    del self._ps_pod_name_to_id[pod_name]
                    relaunch_ps = self._relaunch_deleted_live_ps
            else:
                logger.error("Unknown pod name: %s" % pod_name)
                return

            if self._rendezvous_server:
                self._worker_addrs = self._get_alive_worker_addr()
                self._rendezvous_server.set_worker_hosts(self._worker_addrs)

        if relaunch_worker and worker_id >= 0:
            logger.info("Relaunching worker.")
            new_worker_id = self._next_worker_id_fn()
            with self._lock:
                self._worker_pod_priority[
                    new_worker_id
                ] = self._worker_pod_priority[worker_id]
            self._start_worker(new_worker_id)
        elif relaunch_ps:
            logger.info("Relaunching ps.")
            # Note: the ID and service address for relaunched parameter
            # server are intentionally left unchanged to support fault
            # tolerance.
            self._start_ps(ps_id)

    def check_all_workers_exited(self):
        workers_exited = []
        for (_, _, phase,) in self._worker_pods_ip_phase.values():
            workers_exited.append(
                phase == PodStatus.FAILED or phase == PodStatus.SUCCEEDED
            )
        self.all_workers_exited = all(workers_exited)

    def get_alive_workers(self):
        alive_workers = []
        for pod_name, _, phase in self._worker_pods_ip_phase.values():
            if phase == PodStatus.RUNNING:
                alive_workers.append(pod_name)
        return alive_workers

    def get_worker_pod_ip(self, worker_id):
        if worker_id not in self._worker_pods_ip_phase:
            return None
        _, pod_ip, _ = self._worker_pods_ip_phase[worker_id]
        return pod_ip

    def _get_alive_worker_addr(self):
        alive_workers = self.get_alive_workers()
        worker_addrs = []
        worker_start_times = []
        for pod_name in alive_workers:
            pod = self._k8s_client.get_pod(pod_name)
            worker_start_times.append(pod.status.start_time)
            worker_id = self._worker_pod_name_to_id[pod_name]
            pod_ip = self.get_worker_pod_ip(worker_id)
            worker_addrs.append((pod_name, pod_ip))

        # Sort worker addrs by start time. Then the master will assign
        # the rank according to the order in addrs list.
        worker_addrs = [
            x for _, x in sorted(zip(worker_start_times, worker_addrs))
        ]
        return worker_addrs

    @property
    def ps_addrs(self):
        return self._ps_addrs
