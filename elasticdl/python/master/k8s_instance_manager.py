import copy
import itertools
import threading
from collections import Counter

from elasticdl.python.common import k8s_client as k8s
from elasticdl.python.common.log_utils import default_logger as logger

_SERVICE_ADDR_SEP = ","


class InstanceManager(object):
    def __init__(
        self,
        task_d,
        num_workers=1,
        worker_command=None,
        worker_args=None,
        worker_resource_request="cpu=1,memory=4096Mi",
        worker_resource_limit="cpu=1,memory=4096Mi",
        worker_pod_priority=None,
        num_ps=0,
        ps_command=None,
        ps_args=None,
        ps_resource_request="cpu=1,memory=4096Mi",
        ps_resource_limit="cpu=1,memory=4096Mi",
        ps_pod_priority=None,
        volume=None,
        image_pull_policy=None,
        restart_policy="Never",
        envs=None,
        **kwargs
    ):
        self._num_workers = num_workers
        self._worker_command = worker_command
        self._worker_args = worker_args
        self._worker_resource_request = worker_resource_request
        self._worker_resource_limit = worker_resource_limit
        self._worker_pod_priority = worker_pod_priority

        self._num_ps = num_ps
        self._ps_command = ps_command
        self._ps_args = ps_args
        self._ps_resource_request = ps_resource_request
        self._ps_resource_limit = ps_resource_limit
        self._ps_pod_priority = ps_pod_priority

        self._restart_policy = restart_policy
        self._volume = volume
        self._image_pull_policy = image_pull_policy
        self._envs = envs
        self._task_d = task_d
        self._next_worker_id = itertools.count().__next__

        # Protects followed variables, which are accessed from event_cb.
        self._lock = threading.Lock()
        # worker id to (pod name, phase) mapping
        # phase: None/Pending/Running/Succeeded/Failed/Unknown
        #   None: worker was just launched, haven't received event yet.
        #   Pending: worker pod not started yet
        #   Running: worker pod is running
        #   Succeeded: worker pod finishes all tasks and terminates with
        #       no issue.
        #   Failed: worker pod is killed for some reason
        #   Unknown: unknown
        self._worker_pods_phase = {}
        # pod name to worker id mapping
        self._worker_pod_name_to_id = {}

        self._relaunch_deleted_live_worker = True

        self._ps_pods_phase = {}
        self._ps_pod_name_to_id = {}
        self._relaunch_deleted_live_ps = True

        self._failed_pods = []

        self._k8s_client = k8s.Client(event_callback=self._event_cb, **kwargs)
        self._ps_addrs = self._get_addrs(
            self._num_ps, self._k8s_client.get_ps_service_address
        )
        # TODO: Select a worker address to be used for broadcasting model
        # parameters under allreduce-strategy.
        self._worker_addrs = self._get_addrs(
            self._num_workers, self._k8s_client.get_worker_service_address
        )

    def _start_worker(self, worker_id):
        logger.info("Starting worker: %d" % worker_id)
        with self._lock:
            pod = self._k8s_client.create_worker(
                worker_id=worker_id,
                resource_requests=self._worker_resource_request,
                resource_limits=self._worker_resource_limit,
                pod_priority=self._worker_pod_priority,
                volume=self._volume,
                image_pull_policy=self._image_pull_policy,
                command=self._worker_command,
                args=self._worker_args
                + ["--worker_id", str(worker_id)]
                + ["--ps_addrs", self._ps_addrs],
                restart_policy=self._restart_policy,
                ps_addrs=self._ps_addrs,
                envs=copy.deepcopy(self._envs),
            )
            name = pod.metadata.name
            self._worker_pod_name_to_id[name] = worker_id
            self._worker_pods_phase[worker_id] = (name, None)
            self._k8s_client.create_worker_service(worker_id)

    def _start_ps(self, ps_id):
        logger.info("Starting PS: %d" % ps_id)
        with self._lock:
            pod = self._k8s_client.create_ps(
                ps_id=ps_id,
                resource_requests=self._ps_resource_request,
                resource_limits=self._ps_resource_limit,
                pod_priority=self._ps_pod_priority,
                volume=self._volume,
                image_pull_policy=self._image_pull_policy,
                command=self._ps_command,
                args=self._ps_args + ["--ps_id", str(ps_id)],
                restart_policy=self._restart_policy,
                envs=copy.deepcopy(self._envs),
            )
            name = pod.metadata.name
            self._ps_pod_name_to_id[name] = ps_id
            self._ps_pods_phase[ps_id] = (name, None)
            self._k8s_client.create_ps_service(ps_id)

    def _get_addrs(self, num_addrs, addr_get_fn):
        addrs = []
        for addr_id in range(num_addrs):
            addrs.append(addr_get_fn(addr_id))
        return _SERVICE_ADDR_SEP.join(addrs)

    @staticmethod
    def _update_addr(old_addr, new_addr, addrs, addr_get_fn):
        addrs_list = addrs.split(_SERVICE_ADDR_SEP)
        addrs_list[addrs_list.index(addr_get_fn(old_addr))] = addr_get_fn(
            new_addr
        )
        return _SERVICE_ADDR_SEP.join(addrs_list)

    def update_status(self, status):
        master_name = self._k8s_client.get_master_pod_name()
        self._k8s_client.patch_labels_to_pod(
            master_name, labels_dict={"status": status}
        )

    def start_workers(self):
        for _ in range(self._num_workers):
            self._start_worker(self._next_worker_id())

    def start_parameter_servers(self):
        for i in range(self._num_ps):
            self._start_ps(i)

    def _remove_worker(self, worker_id):
        logger.info("Removing worker: %d", worker_id)
        with self._lock:
            if worker_id not in self._worker_pods_phase:
                logger.error("Unknown worker id: %s" % worker_id)
                return

        # TODO: change _k8s_client to accept pod name instead of worker id.
        self._k8s_client.delete_worker(worker_id)

    def _remove_ps(self, ps_id):
        logger.info("Removing PS: %d", ps_id)
        with self._lock:
            if ps_id not in self._ps_pods_phase:
                logger.error("Unknown PS id: %s" % ps_id)
                return

        self._k8s_client.delete_ps(ps_id)

    def stop_relaunch_and_remove_workers(self):
        with self._lock:
            self._relaunch_deleted_live_worker = False
            for worker_id in self._worker_pods_phase:
                self._k8s_client.delete_worker(worker_id)

    def stop_relaunch_and_remove_all_ps(self):
        with self._lock:
            self._relaunch_deleted_live_ps = False
            for ps_id in self._ps_pods_phase:
                self._k8s_client.delete_ps(ps_id)

    def get_worker_counter(self):
        with self._lock:
            return Counter([v for _, v in self._worker_pods_phase.values()])

    def get_ps_counter(self):
        with self._lock:
            return Counter([v for _, v in self._ps_pods_phase.values()])

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
        phase = evt_obj.status.phase
        logger.info(
            "Got event %s, phase %s for pod: %s" % (evt_type, phase, pod_name)
        )
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
            if pod_name in self._worker_pod_name_to_id:
                worker_id = self._worker_pod_name_to_id.get(pod_name)
                self._worker_pods_phase[worker_id] = (pod_name, phase)
                # Workaround for memory leak issues in tf eager mode.
                # A pod may fail due to OOM from tf eager mode memory leak.
                failed_pod = False
                if evt_type == "MODIFIED" and phase == "Failed":
                    self._failed_pods.append(pod_name)
                    failed_pod = True
                if evt_type == "DELETED" or failed_pod:
                    del self._worker_pods_phase[worker_id]
                    del self._worker_pod_name_to_id[pod_name]
                    self._task_d.recover_tasks(worker_id)

                    # If a deleted pod was not "Succeeded", relaunch a worker.
                    relaunch_worker = (
                        self._relaunch_deleted_live_worker
                        and phase != "Succeeded"
                    )

            elif pod_name in self._ps_pod_name_to_id:
                ps_id = self._ps_pod_name_to_id.get(pod_name)
                self._ps_pods_phase[ps_id] = (pod_name, phase)
                # Workaround for memory leak issues in tf eager mode.
                # A pod may fail due to OOM from tf eager mode memory leak.
                failed_pod = False
                if evt_type == "MODIFIED" and phase == "Failed":
                    self._failed_pods.append(pod_name)
                    failed_pod = True
                if evt_type == "DELETED" or failed_pod:
                    del self._ps_pods_phase[ps_id]
                    del self._ps_pod_name_to_id[pod_name]
                    relaunch_ps = self._relaunch_deleted_live_ps
            else:
                logger.error("Unknown pod name: %s" % pod_name)
                return

        if relaunch_worker and worker_id >= 0:
            logger.info("Relaunching worker.")
            new_worker_id = self._next_worker_id()
            self._start_worker(new_worker_id)
            self._update_addr(
                worker_id,
                new_worker_id,
                self._worker_addrs,
                addr_get_fn=self._k8s_client.get_worker_service_address,
            )
        elif relaunch_ps:
            logger.info("Relaunching ps.")
            # Note: the ID and service address for relaunched parameter
            # server are intentionally left unchanged to support fault
            # tolerance.
            self._start_ps(ps_id)

    @property
    def ps_addrs(self):
        return self._ps_addrs
