import itertools
import logging
import threading

from collections import Counter
from elasticdl.python.elasticdl.master import k8s_client as k8s


class WorkerManager(object):
    def __init__(
            self,
            task_q,
            command,
            args,
            num_worker=1,
            cpu_request="1000m",
            cpu_limit="1000m",
            memory_request="4096Mi",
            memory_limit="4096Mi",
            pod_priority=None,
            mount_path=None,
            volume_name=None,
            image_pull_policy=None,
            restart_policy="OnFailure",
            **kwargs):
        self._logger = logging.getLogger("WorkerManager")
        self._command = command
        self._args = args
        self._num_worker = num_worker
        self._resource_requests = {
            "cpu": cpu_request,
            "memory": memory_request
        }
        self._resource_limits = {
            "cpu": cpu_limit,
            "memory": memory_limit
        }
        self._restart_policy = restart_policy
        self._pod_priority = pod_priority
        self._mount_path = mount_path
        self._volume_name = volume_name
        self._image_pull_policy=image_pull_policy
        self._task_q = task_q
        self._next_worker_id = itertools.count().__next__

        # protects followed variables, which are accessed from event_cb.
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
        self._pods_phase = {}
        # pod name to worker id mapping
        self._pod_name_to_id = {}

        self._relaunch_deleted_live_worker = True

        self._k8s_client = k8s.Client(
            event_callback=self._event_cb, **kwargs
        )

    def set_relaunch_deleted_live_worker(self, val):
        self._relaunch_deleted_live_worker = bool(val)

    def _start_worker(self, worker_id):
        self._logger.info("Starting worker: %d" % worker_id)
        with self._lock:
            pod = self._k8s_client.create_worker(
                worker_id,
                self._resource_requests,
                self._resource_limits,
                self._pod_priority,
                self._mount_path,
                self._volume_name,
                self._image_pull_policy,
                command=self._command,
                args=self._args + ["--worker_id", str(worker_id)],
                restart_policy=self._restart_policy,
            )
            name = pod.metadata.name
            self._pod_name_to_id[name] = worker_id
            self._pods_phase[worker_id] = (name, None)

    def start_workers(self):
        for i in range(self._num_worker):
            self._start_worker(self._next_worker_id())

    def _remove_worker(self, worker_id):
        with self._lock:
            if worker_id not in self._pods_phase:
                self._logger.error("Unknown worker id: %s" % worker_id)
                return

        # TODO: change _k8s_client to accept pod name instead of worker id.
        self._k8s_client.delete_worker(worker_id)

    def stop_relaunch_and_remove_workers(self):
        with self._lock:
            self._relaunch_deleted_live_worker = False
            for worker_id in self._pods_phase:
                self._k8s_client.delete_worker(worker_id)

    def get_counters(self):
        with self._lock:
            return Counter([v for _, v in self._pods_phase.values()])

    def _event_cb(self, event):
        evt_obj = event.get("object")
        evt_type = event.get("type")
        if not evt_obj or not evt_type:
            self._logger.error("Event doesn't have object or type: %s" % event)
            return

        pod_name = evt_obj.metadata.name
        relaunch = False
        with self._lock:
            worker_id = self._pod_name_to_id.get(pod_name)
            if worker_id is None:
                self._logger.error("Unknown pod name: %s" % pod_name)
                return

            phase = evt_obj.status.phase
            self._pods_phase[worker_id] = (pod_name, phase)
            if evt_type == "DELETED":
                del self._pods_phase[worker_id]
                del self._pod_name_to_id[pod_name]
                self._task_q.recover_tasks(worker_id)

                # If the pod being deleted was 'Running' or 'Pending'. Launch
                # a worker.
                relaunch = self._relaunch_deleted_live_worker and phase in [
                        "Running", "Pending"]
        if relaunch:
            self._start_worker(self._next_worker_id())
