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
import json
import os
import re
import threading
import time
from collections import Counter

from kubernetes.client import V1EnvVar

from elasticai_api.common.constants import PodEnv, WorkerEnv
from elasticdl.python.common import k8s_client as k8s
from elasticdl.python.common.constants import (
    PodManagerStatus,
    PodStatus,
    WorkerMemoryConfig,
)
from elasticdl.python.common.k8s_client import PodType
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.common.model_utils import get_dict_from_params_str
from elasticdl.python.master.pod_event_callbacks import ClusterContext
from elasticdl.python.master.pod_info import (
    TypedPodConfig,
    init_pod_info,
    set_critical_pod,
    set_worker_pod_priority,
)
from elasticdl.python.master.pod_state import get_pod_state_flow
from elasticdl.python.master.resource_monitor import WorkerResourceMonitor
from elasticdl.python.master.worker_sync import WorkerSyncObjects
from elasticdl_client.common.args import parse_envs
from elasticdl_client.common.constants import (
    BashCommandTemplate,
    ClusterSpecConfig,
    DistributionStrategy,
)
from elasticdl_client.common.k8s_client import (
    ELASTICDL_REPLICA_INDEX_KEY,
    ELASTICDL_REPLICA_TYPE_KEY,
)

_SERVICE_ADDR_SEP = ","
_MIN_OOM_RELAUNCH_COUNT = 2


def _get_ps_addrs(num_addrs, addr_get_fn):
    """
    Get `num_addrs` addresses and then concatenate
    them to a comma separated string.
    """
    addrs = []
    for addr_id in range(num_addrs):
        addrs.append(addr_get_fn(PodType.PS, addr_id))
    return _SERVICE_ADDR_SEP.join(addrs)


def _is_killed_pod(evt_obj):
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


def _is_oom_pod(evt_obj):
    return (
        evt_obj.status.container_statuses
        and evt_obj.status.container_statuses[0].state.terminated
        and evt_obj.status.container_statuses[0].state.terminated.exit_code
        == 137
        and evt_obj.status.container_statuses[0].state.terminated.reason
        == "OOMKilled"
    )


def _get_start_running_time_stamp(pod_status_obj):
    if (
        pod_status_obj.container_statuses
        and pod_status_obj.container_statuses[0].state
        and pod_status_obj.container_statuses[0].state.running
    ):
        return pod_status_obj.container_statuses[0].state.running.started_at

    return None


def get_image_cluster_spec(cluster_spec):
    if cluster_spec:
        filename = os.path.basename(cluster_spec)
        image_cluster_spec = os.path.join(
            ClusterSpecConfig.CLUSTER_SPEC_DIR, filename
        )
        return image_cluster_spec
    return cluster_spec


def build_environment_variables(args):
    env = []

    env_dict = parse_envs(args.envs)
    for key, value in env_dict.items():
        env.append(V1EnvVar(name=key, value=value))

    master_ip = os.getenv("MY_POD_IP", "localhost")
    master_addr = "%s:%d" % (master_ip, args.port)
    env.append(V1EnvVar(name=WorkerEnv.MASTER_ADDR, value=master_addr))
    env.append(
        V1EnvVar(name=WorkerEnv.WORKER_NUM, value=str(args.num_workers))
    )

    if args.populate_env_names:
        regex = re.compile(args.populate_env_names)
        for key, value in os.environ.items():
            if regex.fullmatch(key):
                env.append(V1EnvVar(name=key, value=value))

    return env


def is_tfv1_ps_strategy_custom_training(
    need_elasticdl_job_service, need_tf_config, distribution_strategy
):
    if (
        not need_elasticdl_job_service
        and need_tf_config
        and distribution_strategy == DistributionStrategy.PARAMETER_SERVER
    ):
        return True

    return False


def is_huge_memory(typed_pod_config):
    worker_mem = typed_pod_config.get_typed_resource_config(
        PodType.WORKER
    ).get_memory_mi()
    worker_num = typed_pod_config.get_typed_pod_num(PodType.WORKER)
    return worker_num * worker_mem > WorkerMemoryConfig.HUGE_RESOURCE_THRESHOLD


def should_launch_worker_after_ps_running(is_tfv1_ps, is_huge_memory, args):
    if args.launch_worker_after_ps_running == "default":
        return is_tfv1_ps and is_huge_memory
    elif args.launch_worker_after_ps_running == "on":
        return args.num_ps_pods > 0
    else:
        return False


def get_critical_worker_index(is_tfv1_ps, args):
    critical_worker_index = {}
    if args.critical_worker_index == "default":
        # for default, worker0 is critical if PS strategy with custom training
        if is_tfv1_ps:
            critical_worker_index[0] = 0
    elif args.critical_worker_index != "none":
        for pod_relaunch_conf in args.critical_worker_index.split(","):
            # The conf is "pod_index:relaunch_times"
            pod_relaunch = pod_relaunch_conf.strip().split(":")
            critical_worker_index[int(pod_relaunch[0])] = int(pod_relaunch[1])

    return critical_worker_index


def create_pod_manager(args):
    pod_manager = None

    if args.num_workers:
        assert args.worker_image, "Worker image cannot be empty"

        env = build_environment_variables(args)
        kwargs = get_dict_from_params_str(args.aux_params)
        disable_relaunch = kwargs.get("disable_relaunch", False)
        cluster_spec = get_image_cluster_spec(args.cluster_spec)

        # relaunch on worker failure for PS or custom strategy
        if (
            args.distribution_strategy == DistributionStrategy.PARAMETER_SERVER
            or args.distribution_strategy == DistributionStrategy.CUSTOM
        ):
            relaunch_on_worker_failure = args.relaunch_on_worker_failure
        else:
            relaunch_on_worker_failure = 0

        is_tfv1_ps = is_tfv1_ps_strategy_custom_training(
            args.need_elasticdl_job_service,
            args.need_tf_config,
            args.distribution_strategy,
        )

        critical_worker_index = get_critical_worker_index(is_tfv1_ps, args)

        # Custom distribution strategy does not exit if there are pending pods
        wait_pending_relaunch = (
            args.distribution_strategy == DistributionStrategy.CUSTOM
        )

        typed_pod_config = TypedPodConfig()
        typed_pod_config.add_typed_pod_config(
            PodType.WORKER,
            args.num_workers,
            args.worker_resource_request,
            args.worker_resource_limit,
            args.worker_pod_priority,
            args.worker_image if args.worker_image else args.image_name,
        )

        typed_pod_config.add_typed_pod_config(
            PodType.PS,
            args.num_ps_pods,
            args.ps_resource_request,
            args.ps_resource_limit,
            args.ps_pod_priority,
            args.ps_image if args.ps_image else args.image_name,
        )

        # Keep the same as the worker.
        typed_pod_config.add_typed_pod_config(
            PodType.EVALUATOR,
            args.num_evaluators,
            args.evaluator_resource_request,
            args.evaluator_resource_limit,
            args.evaluator_pod_priority,
            args.worker_image if args.worker_image else args.image_name,
        )

        typed_pod_config.add_typed_pod_config(
            PodType.CHIEF,
            1,
            args.chief_resource_request,
            args.chief_resource_limit,
            args.chief_pod_priority,
            args.worker_image if args.worker_image else args.image_name,
        )

        huge_memory_configured = is_huge_memory(typed_pod_config)
        launch_worker_after_ps_running = should_launch_worker_after_ps_running(
            is_tfv1_ps, huge_memory_configured, args
        )
        enable_automate_memory = (
            args.enable_automate_memory
            and huge_memory_configured
            and launch_worker_after_ps_running
        )

        pod_manager = PodManager(
            typed_pod_config=typed_pod_config,
            job_name=args.job_name,
            image_name=args.image_name,
            namespace=args.namespace,
            volume=args.volume,
            image_pull_policy=args.image_pull_policy,
            restart_policy=args.restart_policy,
            cluster_spec=cluster_spec,
            cluster_spec_json=args.cluster_spec_json,
            envs=env,
            need_tf_config=args.need_tf_config,
            disable_relaunch=disable_relaunch,
            log_file_path=args.log_file_path,
            need_elasticdl_job_args=args.need_elasticdl_job_service,
            relaunch_on_worker_failure=relaunch_on_worker_failure,
            ps_is_critical=args.ps_is_critical,
            critical_worker_index=critical_worker_index,
            wait_pending_relaunch=wait_pending_relaunch,
            ps_relaunch_max_num=args.ps_relaunch_max_num,
            distribution_strategy=args.distribution_strategy,
            launch_worker_after_ps_running=launch_worker_after_ps_running,
            enable_automate_memory=enable_automate_memory,
        )

    return pod_manager


class PodManager(object):
    def __init__(
        self,
        typed_pod_config=TypedPodConfig(),
        volume=None,
        image_pull_policy=None,
        restart_policy="Never",
        envs=None,
        need_tf_config=False,
        disable_relaunch=False,
        log_file_path=None,
        need_elasticdl_job_args=False,
        relaunch_on_worker_failure=0,
        ps_is_critical=True,
        critical_worker_index={},
        wait_pending_relaunch=False,
        ps_relaunch_max_num=1,
        distribution_strategy=None,
        launch_worker_after_ps_running=False,
        enable_automate_memory=True,
        **kwargs
    ):
        self._typed_pod_config = typed_pod_config
        self._restart_policy = restart_policy
        self._volume = volume
        self._image_pull_policy = image_pull_policy
        self._envs = envs

        self._log_file_path = log_file_path
        self._need_tf_config = need_tf_config
        self._need_elasticdl_job_args = need_elasticdl_job_args
        self._relaunch_on_worker_failure = relaunch_on_worker_failure
        self._wait_pending_relaunch = wait_pending_relaunch
        self._distribution_strategy = distribution_strategy
        self._launch_worker_after_ps_running = launch_worker_after_ps_running
        self._enable_automate_memory = enable_automate_memory
        self._worker_sync_objs = WorkerSyncObjects(pod_manager=self)
        self._start_launch_waiting_workers_time = time.time()
        self._stop_launch_worker_for_ps = False

        # Protects followed variables, which are accessed from event_cb.
        self._lock = threading.Lock()

        self._init_pod_status()
        set_critical_pod(self.pod_info, ps_is_critical, critical_worker_index)
        set_worker_pod_priority(
            self.pod_info,
            typed_pod_config.get_typed_resource_config(
                PodType.WORKER
            ).priority,
        )

        if disable_relaunch:
            self._k8s_client = k8s.Client(**kwargs)
        else:
            self._k8s_client = k8s.Client(
                event_callback=self._event_cb,
                periodic_call_func=self._process_worker,
                **kwargs
            )
        self._num_ps = typed_pod_config.get_typed_pod_num(PodType.PS)
        self._num_workers = typed_pod_config.get_typed_pod_num(PodType.WORKER)
        self._num_evaluators = typed_pod_config.get_typed_pod_num(
            PodType.EVALUATOR
        )
        self._num_chief = typed_pod_config.get_typed_pod_num(
            PodType.CHIEF
        )
        self._ps_addrs = _get_ps_addrs(
            self._num_ps, self._k8s_client.get_service_address
        )
        self._worker_addrs = []
        self._worker_command = None
        self._worker_args = None
        self._ps_command = None
        self._ps_args = None
        self._pod_event_callbacks = []
        self._oom_relaunch_count = 0
        self._max_oom_relaunch_count = max(
            _MIN_OOM_RELAUNCH_COUNT, self._num_workers
        )

        self._worker_resource_monitor = WorkerResourceMonitor()
        self.wait_chief_worker_execution = True
        self.data_shard_service_enabled = False

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
        self.start_chief()
        self.start_workers()
        self.start_evaluators()
        self.update_status(PodManagerStatus.RUNNING)

    def add_pod_event_callback(self, pod_event_callback):
        self._pod_event_callbacks.append(pod_event_callback)

    def _init_pod_status(self):
        self.pod_info = init_pod_info(
            self._relaunch_on_worker_failure, self._typed_pod_config
        )

        # worker ids for the pods which are not created.
        # We will try multiple times in the background to create pods
        # using the ids in the list until success.
        self._not_created_workers = []

        # worker and eval ids for pods that should be created
        # after all ps are running.
        self._workers_waiting_ps_running = []

        # ps pod ids that are deleted and waiting for relaunch
        self._deleted_ps_pod_ids = []

        self._relaunch_pod = True
        self._pending_relaunch_count = 0
        self._init_pod_id_iter()

    def _init_pod_id_iter(self):
        self._pod_id_iter = {}
        for pod_type in self._typed_pod_config.get_pod_types():
            self._pod_id_iter[pod_type] = itertools.count(
                self._typed_pod_config.get_typed_pod_num(pod_type)
            )

    def _get_next_pod_id(self, pod_type):
        return next(self._pod_id_iter[pod_type])

    def _process_worker(self):
        need_process = True
        not_created_workers = []

        if self.data_shard_service_enabled:
            wait_chief_worker_timeout = (
                time.time() - self._start_launch_waiting_workers_time
                > WorkerMemoryConfig.WAIT_CHIEF_WORKER_TIMEOUT_SECS
            )
        else:
            wait_chief_worker_timeout = (
                time.time() - self._start_launch_waiting_workers_time
                > WorkerMemoryConfig.WAIT_DATA_SHARD_SERVICE_CREATION_SECS
            )

        if wait_chief_worker_timeout:
            self.wait_chief_worker_execution = False
        # First launch the pods which need not to wait the chief worker.
        while need_process and self._not_created_workers:
            pod_type, worker_id = self._not_created_workers.pop(0)
            pod_info = self.pod_info[pod_type][worker_id]
            if (
                self._enable_automate_memory
                and pod_info.type == PodType.WORKER
                and not pod_info.is_recovered_oom
                and pod_info.original_index != 0
            ):
                if self.wait_chief_worker_execution:
                    if not not_created_workers:
                        logger.info(
                            "Cannot start workers until the chief "
                            "worker starts training"
                        )
                    not_created_workers.append((pod_type, worker_id))
                    continue
                elif not wait_chief_worker_timeout:
                    recommended_mem = self._get_recommended_memory(pod_type)
                    if recommended_mem > 0:
                        pod_info = self.pod_info[pod_type][worker_id]
                        pod_info.set_memory(recommended_mem)
            # Try to create a worker pod with id as worker_id
            if not self._stop_launch_worker_for_ps or pod_info.is_critical_pod:
                need_process = self._start_typed_worker(pod_type, worker_id)
        with self._lock:
            self._not_created_workers.extend(not_created_workers)

    def _get_recommended_memory(self, pod_type):
        configured_mem = self._typed_pod_config.get_typed_resource_config(
            pod_type
        ).get_memory_mi()
        used_memory = self._worker_resource_monitor.get_worker_memory()
        logger.info(
            "The memory usage of the chief worker is {}Mi".format(used_memory)
        )
        if used_memory > 0:
            recommended_mem = min(
                used_memory * WorkerMemoryConfig.ADJUSTMENT_FACTOR,
                used_memory + WorkerMemoryConfig.MAX_INCREMENTAL_MEMORY,
            )
            recommended_mem = max(
                recommended_mem, WorkerMemoryConfig.MIN_MEMORY,
            )
            used_memory = min(recommended_mem, configured_mem)
        return used_memory

    def _start_pod(self, pod_type, id):
        if pod_type == PodType.PS:
            return self._start_ps(id)
        if pod_type == PodType.WORKER or pod_type == PodType.EVALUATOR:
            return self._start_typed_worker(pod_type, id)

    def _start_typed_worker(self, pod_type, pod_id):
        logger.info("Starting %s : %d" % (pod_type, pod_id))
        ps_node_argument = (
            " --{} {}".format("ps_addrs", self._ps_addrs)
            if self._need_elasticdl_job_args
            else None
        )
        job_command = self._complement_job_command(
            self._worker_args, ps_node_argument
        )
        worker_args = [self._worker_args[0], job_command]
        envs = copy.deepcopy(self._envs)
        envs.append(V1EnvVar(name=WorkerEnv.WORKER_ID, value=str(pod_id)))
        if_relaunch_str = (
            "true"
            if pod_id != self.pod_info[pod_type][pod_id].original_index
            else "false"
        )
        envs.append(
            V1EnvVar(name=PodEnv.RELAUNCHED_POD, value=if_relaunch_str)
        )
        need_create_service = False
        need_patch_service = False
        original_index = pod_id
        if self._need_tf_config:
            original_index = self.pod_info[pod_type][pod_id].original_index
            tf_config = self.get_tf_config_data(pod_type, original_index)
            envs.append(
                V1EnvVar(name="TF_CONFIG", value=json.dumps(tf_config))
            )
            if original_index == pod_id:
                need_create_service = True
            else:
                need_patch_service = True
        resource_config = self.pod_info[pod_type][pod_id].resource_config
        with self._lock:
            pod = self._k8s_client.create_typed_pod(
                pod_type,
                pod_id,
                resource_requests=resource_config.resource_requests,
                resource_limits=resource_config.resource_limits,
                pod_priority=resource_config.priority,
                termination_period=1,
                volume=self._volume,
                image_pull_policy=self._image_pull_policy,
                command=self._worker_command,
                args=worker_args,
                restart_policy=self._restart_policy,
                ps_addrs=self._ps_addrs,
                envs=envs,
                image_name=resource_config.image_name,
            )
            if pod is None:
                self._not_created_workers.append((pod_type, pod_id))
                return False
            # create or patch worker service
            if need_create_service:
                self._k8s_client.create_service(pod_type, pod_id)
            if need_patch_service:
                self._k8s_client.patch_service(
                    pod_type, original_index, pod_id
                )

            return True

    def _complement_job_command(self, pod_args, ps_node_argument):
        # pod_args has 2 strings. The first string is "-c" and
        # the second string is the shell command to run, like
        # ["-c", "python -m main --minibatch_size 64"]
        job_command = pod_args[1]
        if ps_node_argument:
            job_command += ps_node_argument
        if self._log_file_path:
            job_command = BashCommandTemplate.REDIRECTION.format(
                job_command, self._log_file_path
            )
        job_command += " ".join(pod_args[2:])
        job_command = BashCommandTemplate.SET_PIPEFAIL + job_command
        return job_command

    def _start_ps(self, ps_id):
        logger.info("Starting PS: %d" % ps_id)
        ps_node_argument = (
            " --{} {}".format("ps_id", ps_id)
            if self._need_elasticdl_job_args
            else None
        )
        bash_command = self._complement_job_command(
            self._ps_args, ps_node_argument
        )
        ps_args = [self._ps_args[0], bash_command]
        original_index = self.pod_info[PodType.PS][ps_id].original_index
        while True:
            with self._lock:
                pod = self._create_ps_pod(ps_id, original_index, ps_args)
                if pod:
                    if ps_id == original_index:
                        self._k8s_client.create_service(PodType.PS, ps_id)
                    else:  # patch service
                        self._k8s_client.patch_service(
                            PodType.PS, original_index, ps_id
                        )
                    break
            # TODO: should we fail the job when ps pods fail to
            #       create for a long time?
            logger.error(
                "Creating PS fails and will try again."
                "ps_id: {}, ps_args: {}.".format(ps_id, ps_args)
            )
            time.sleep(15)

    def get_tf_config_data(self, type_key, index_key):
        cluster_dict = {}
        if self._num_ps > 0:
            cluster_dict[PodType.PS] = []
            for ps_id in range(self._num_ps):
                cluster_dict[PodType.PS].append(
                    self._k8s_client.get_service_address(PodType.PS, ps_id)
                )
        if self._num_workers > 0:
            cluster_dict[PodType.WORKER] = []
            for worker_id in range(self._num_workers):
                cluster_dict[PodType.WORKER].append(
                    self._k8s_client.get_service_address(
                        PodType.WORKER, worker_id
                    )
                )
        if self._num_evaluators > 0:
            cluster_dict[PodType.EVALUATOR] = []
            for worker_id in range(self._num_evaluators):
                cluster_dict[PodType.EVALUATOR].append(
                    self._k8s_client.get_service_address(
                        PodType.EVALUATOR, worker_id
                    )
                )
        if self._num_chief > 0:
            cluster_dict[PodType.CHIEF] = []
            for worker_id in range(self._num_chief):
                cluster_dict[PodType.CHIEF].append(
                    self._k8s_client.get_service_address(
                        PodType.CHIEF, worker_id
                    )
                )

        task_dict = {}
        task_dict["type"] = type_key
        task_dict["index"] = index_key
        return {"cluster": cluster_dict, "task": task_dict}

    def _create_ps_pod(self, ps_id, original_index, ps_args):
        envs = copy.deepcopy(self._envs)

        envs.append(
            V1EnvVar(
                name=PodEnv.RELAUNCHED_POD,
                value="false" if ps_id == original_index else "true",
            )
        )
        if self._need_tf_config:
            tf_config = self.get_tf_config_data(PodType.PS, original_index)
            envs.append(
                V1EnvVar(name="TF_CONFIG", value=json.dumps(tf_config))
            )
        resource_config = self.pod_info[PodType.PS][ps_id].resource_config
        return self._k8s_client.create_typed_pod(
            PodType.PS,
            ps_id,
            resource_requests=resource_config.resource_requests,
            resource_limits=resource_config.resource_limits,
            pod_priority=resource_config.priority,
            volume=self._volume,
            image_pull_policy=self._image_pull_policy,
            command=self._ps_command,
            args=ps_args,
            restart_policy=self._restart_policy,
            envs=envs,
            image_name=resource_config.image_name,
        )

    def update_status(self, status):
        master_name = self._k8s_client.get_master_pod_name()
        self._k8s_client.patch_labels_to_pod(
            master_name, labels_dict={"status": status}
        )

    def start_workers(self):
        for i in range(self._num_workers):
            self._start_typed_worker(PodType.WORKER, i)
            if i == 0:
                self._worker_resource_monitor.set_chief_worker_name(
                    self._k8s_client.get_pod_name(PodType.WORKER, i)
                )
            if self._launch_worker_after_ps_running:
                # only launch worker0 and put others into a waiting list
                with self._lock:
                    for j in range(1, self._num_workers):
                        self._workers_waiting_ps_running.append(
                            (PodType.WORKER, j)
                        )
                break

    def start_parameter_servers(self):
        for i in range(self._num_ps):
            self._start_ps(i)

    def start_evaluators(self):
        if self._launch_worker_after_ps_running:
            with self._lock:
                for i in range(self._num_evaluators):
                    self._workers_waiting_ps_running.append(
                        (PodType.EVALUATOR, i)
                    )
            return

        for i in range(self._num_evaluators):
            self._start_typed_worker(PodType.EVALUATOR, i)

    def start_chief(self):
        if self._launch_worker_after_ps_running:
            with self._lock:
                for i in range(self._num_chief):
                    self._workers_waiting_ps_running.append(
                        (PodType.CHIEF, i)
                    )
            return

        for i in range(self._num_chief):
            self._start_typed_worker(PodType.CHIEF, i)

    def remove_worker(self, worker_id):
        logger.info("Removing worker: %d", worker_id)
        with self._lock:
            if worker_id not in self.pod_info[PodType.WORKER] or self.pod_info[
                PodType.WORKER
            ][worker_id] in [PodStatus.DELETED, PodStatus.INITIAL]:
                logger.error("Unknown deletable worker id: %s" % worker_id)
                return

        # TODO: change _k8s_client to accept pod name instead of worker id.
        self._k8s_client.delete_typed_pod(PodType.WORKER, worker_id)

    def remove_running_ps_training_pods(self):
        """Remove all running PServer nodes and non-critical workers"""
        if not is_tfv1_ps_strategy_custom_training(
            self._need_elasticdl_job_args,
            self._need_tf_config,
            self._distribution_strategy,
        ):
            return

        self._stop_launch_worker_for_ps = True
        # Allow to relaunch the evaluator if there is no alive worker.
        self._wait_pending_relaunch = True

        for info in self.pod_info[PodType.WORKER].values():
            if not info.is_critical_pod and info.status in [
                PodStatus.RUNNING,
                PodStatus.PENDING,
            ]:
                info.relaunchable = False
                logger.info(
                    "Remove the pod {} after the worker-0 completed".format(
                        info.name
                    )
                )
                self._k8s_client.delete_pod(info.name)

        for info in self.pod_info[PodType.PS].values():
            if info.status in [PodStatus.RUNNING, PodStatus.PENDING]:
                info.is_critical_pod = False
                info.relaunchable = False
                logger.info(
                    "Remove the pod {} after the worker-0 completed".format(
                        info.name
                    )
                )
                self._k8s_client.delete_pod(info.name)

    def stop_relaunch_and_remove_all_pods(self):
        self._relaunch_pod = False
        with self._lock:
            for pod_type in self.pod_info.keys():
                for info in self.pod_info[pod_type].values():
                    if info.status in [PodStatus.RUNNING, PodStatus.PENDING]:
                        logger.info("Remove the pod {}".format(info.name))
                        info.is_critical_pod = False
                        self._k8s_client.delete_pod(info.name)

    def get_pod_counter(self, pod_type):
        with self._lock:
            return Counter(
                [
                    pod_info.status
                    for pod_info in self.pod_info[pod_type].values()
                ]
            )

    def _should_relaunch(self, pod_info, matched_pod_state_flow, evt_obj):
        should_relaunch = (
            matched_pod_state_flow.should_relaunch
            and self._relaunch_pod
            and (
                pod_info.relaunchable
                or self.is_deleted_ps_pod_for_relaunch(pod_info)
            )
        )
        if should_relaunch and not _is_killed_pod(evt_obj):
            if pod_info.relaunch_count < pod_info.max_relaunch_count:
                pod_info.inc_relaunch_count()
            else:
                should_relaunch = False

        if should_relaunch and _is_oom_pod(evt_obj):
            if self._oom_relaunch_count < self._max_oom_relaunch_count:
                self._oom_relaunch_count += 1
                resource = self._typed_pod_config.get_typed_resource_config(
                    pod_info.type
                )
                pod_info.is_recovered_oom = True
                pod_info.increment_memory(resource.get_memory_mi())
            else:
                logger.warn("The relaunch count for OOM has been exhausted.")
                should_relaunch = False

        return should_relaunch

    def _event_cb(self, event):
        evt_obj = event.get("object")
        evt_type = event.get("type")
        if not evt_obj or not evt_type:
            logger.error("Event doesn't have object or type: %s" % event)
            return

        if evt_obj.kind != "Pod":
            # We only care about pod related events
            return

        pod_type = evt_obj.metadata.labels[ELASTICDL_REPLICA_TYPE_KEY]

        if pod_type == PodType.MASTER:
            # No need to care about master pod
            return

        pod_name = evt_obj.metadata.name
        pod_ip = evt_obj.status.pod_ip
        node_ip = evt_obj.status.host_ip
        phase = evt_obj.status.phase
        pod_start_time = _get_start_running_time_stamp(evt_obj.status)

        pod_id = int(evt_obj.metadata.labels[ELASTICDL_REPLICA_INDEX_KEY])
        cur_pod_info = self.pod_info[pod_type][pod_id]

        # For the given pod id, check whether it meets
        # the state change condition
        with self._lock:
            pod_state = cur_pod_info.status
            matched_pod_state_flow = get_pod_state_flow(
                pod_state, evt_type, phase
            )
            # If there is no matched state change, return directly
            # If the pod has been succeed, return directly
            if (
                matched_pod_state_flow is None
                or matched_pod_state_flow.from_status == PodStatus.SUCCEEDED
            ):
                return

            # Update the pod status for pod_info
            new_status = matched_pod_state_flow.to_status
            cur_pod_info.update_info(
                name=pod_name,
                pod_ip=pod_ip,
                node_ip=node_ip,
                status=new_status,
                start_time=pod_start_time,
            )
            self._process_pod_events(matched_pod_state_flow, cur_pod_info)

            should_relaunch = self._should_relaunch(
                cur_pod_info, matched_pod_state_flow, evt_obj
            )
            if should_relaunch and self._wait_pending_relaunch:
                self._pending_relaunch_count += 1

        logger.info(
            "{} status change: {} to {}, by evt_type {}, phase {}".format(
                pod_name, pod_state, new_status, evt_type, phase
            )
        )

        if should_relaunch:
            self._relaunch_typed_pod(cur_pod_info, pod_type)

    def _relaunch_typed_pod(self, cur_pod_info, pod_type):
        logger.info("Relaunch the pod: {}".format(cur_pod_info.name))
        self.metadata_collection.add_relaunch_info(
            [pod_type + str(cur_pod_info.id)]
        )

        with self._lock:
            new_id = self._get_next_pod_id(pod_type)
            self.pod_info[pod_type][
                new_id
            ] = cur_pod_info.get_relaunch_pod_info(new_id)
        self._start_pod(pod_type, new_id)
        if self.is_deleted_ps_pod_for_relaunch(cur_pod_info):
            with self._lock:
                self._deleted_ps_pod_ids.remove(cur_pod_info.id)
        if self._wait_pending_relaunch:
            with self._lock:
                self._pending_relaunch_count -= 1

    def _process_pod_events(self, matched_pod_state_flow, pod_info):
        cluster_context = ClusterContext(pod_manager=self)
        if matched_pod_state_flow.to_status == PodStatus.RUNNING:
            [
                callback.on_pod_started(pod_info, cluster_context)
                for callback in self._pod_event_callbacks
            ]
        elif matched_pod_state_flow.to_status == PodStatus.SUCCEEDED:
            [
                callback.on_pod_succeeded(pod_info, cluster_context)
                for callback in self._pod_event_callbacks
            ]
        elif matched_pod_state_flow.to_status == PodStatus.FAILED:
            [
                callback.on_pod_failed(pod_info, cluster_context)
                for callback in self._pod_event_callbacks
            ]
        elif (
            matched_pod_state_flow.from_status != PodStatus.FAILED
            and matched_pod_state_flow.from_status != PodStatus.SUCCEEDED
            and matched_pod_state_flow.to_status == PodStatus.DELETED
        ):
            [
                callback.on_pod_deleted(pod_info, cluster_context)
                for callback in self._pod_event_callbacks
            ]
        if matched_pod_state_flow.from_status == PodStatus.RUNNING:
            self._worker_sync_objs.non_running_worker_update(pod_info.id)

    @property
    def all_workers_and_evaluators_exited(self):
        worker_counter = self.get_pod_counter(PodType.WORKER)
        evaluator_counter = self.get_pod_counter(PodType.EVALUATOR)
        counter = worker_counter + evaluator_counter

        # At start, there may be no launched worker.
        if len(counter) == 1 and PodStatus.INITIAL in counter:
            return False

        all_exited = (
            PodStatus.RUNNING not in counter
            and PodStatus.PENDING not in counter
        )

        with self._lock:
            if (
                self._wait_pending_relaunch
                and self._pending_relaunch_count > 0
            ):
                all_exited = False

        return all_exited

    @property
    def all_workers_failed(self):
        counter = self.get_pod_counter(PodType.WORKER)
        if len(counter) == 1 and PodStatus.INITIAL in counter:
            return False

        all_failed = all(
            [
                status
                in [PodStatus.FAILED, PodStatus.DELETED, PodStatus.INITIAL]
                for status in counter
            ]
        )

        return all_failed

    @property
    def all_ps_running(self):
        running_ps = [
            pod_info.id
            for pod_info in self.pod_info[PodType.PS].values()
            if pod_info.status == PodStatus.RUNNING
        ]
        return len(running_ps) == self._num_ps

    @property
    def chief_worker_running(self):
        """The chief worker with id=0 is responsible to initialize
        variables in TensorFlow 1.x PS strategy"""
        chief_worker_info = self.pod_info[PodType.WORKER].get(0, None)
        if chief_worker_info:
            return chief_worker_info.status == PodStatus.RUNNING
        return False

    def launch_waiting_workers(self):
        if (
            self._workers_waiting_ps_running
            and self.all_ps_running
            and self.chief_worker_running
        ):
            self._start_launch_waiting_workers_time = time.time()
            self._not_created_workers += self._workers_waiting_ps_running
            self._workers_waiting_ps_running = []
            logger.info("Launching workers after all ps running")

    def get_running_pod_ids(self, pod_type):
        with self._lock:
            return [
                info.id
                for info in self.pod_info[pod_type].values()
                if info.status == PodStatus.RUNNING
            ]

    def get_alive_workers(self):
        with self._lock:
            return [
                info
                for info in self.pod_info[PodType.WORKER].values()
                if info.status == PodStatus.RUNNING
            ]

    def get_alive_worker_id_addr(self):
        alive_workers = self.get_alive_workers()
        alive_workers.sort(key=lambda pod_info: pod_info.start_time)
        return [(info.id, info.pod_ip) for info in alive_workers]

    def get_worker_pod_ip(self, worker_id):
        with self._lock:
            if worker_id in self.pod_info[PodType.WORKER]:
                return self.pod_info[PodType.WORKER][worker_id].pod_ip
        return None

    def get_pod_infos(self, pod_type, pod_statuses):
        with self._lock:
            return [
                info
                for info in self.pod_info[pod_type].values()
                if info.status in pod_statuses
            ]

    @property
    def ps_addrs(self):
        return self._ps_addrs

    def is_deleted_ps_pod_for_relaunch(self, pod_info):
        return (
            pod_info.type == PodType.PS
            and pod_info.id in self._deleted_ps_pod_ids
        )

    def query_relaunch_ps_pod(self):
        if self.slow_pod_relauncher is None:
            return False, []
        return self.slow_pod_relauncher.query_relaunch_ps_pod()

    def relaunch_slow_ps(self):
        if self.slow_pod_relauncher:
            self.slow_pod_relauncher.relaunch_slow_ps()

    def worker_sync(self, sync_name, worker_id):
        return self._worker_sync_objs.worker_sync(sync_name, worker_id)

    def wait_worker_sync(self, sync_name, notify):
        return self._worker_sync_objs.wait_worker_sync(sync_name, notify)

    def delete_worker_sync(self, sync_name, delete_all=False):
        self._worker_sync_objs.delete_worker_sync(sync_name, delete_all)

    def set_worker_resource(self, memory, cpu_percent):
        self._worker_resource_monitor.set_worker_resource(memory, cpu_percent)
