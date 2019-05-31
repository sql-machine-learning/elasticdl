import logging
import threading

from collections import Counter
from elasticdl.python.elasticdl.master import k8s


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
        self._pod_priority = pod_priority
        self._mount_path = mount_path
        self._volume_name = volume_name
        self._image_pull_policy=image_pull_policy
        self._task_q = task_q

        # protects followed variables, which are accessed from event_cb.
        self._lock = threading.Lock()
        # pod name to phase mapping
        # phase: Pending/Running/Succeeded/Failed/Unknown
        #   Pending: worker pod not started yet
        #   Running: worker pod is running
        #   Succeeded: worker pod finishes all jobs and terminates with no issue.
        #   Failed: worker pod is killed for some reason
        #   Unknown: unknown
        self._pods_phase = {}

        self._k8s_client = k8s.Client(
            event_callback=self._event_cb, **kwargs
        )

    def start_workers(self, restart_policy="OnFailure"):
        for i in range(self._num_worker):
            self._logger.info("Starting worker: %d" % i)
            self._k8s_client.create_worker(
                i,
                self._resource_requests,
                self._resource_limits,
                self._pod_priority,
                self._mount_path,
                self._volume_name,
                self._image_pull_policy,
                command=self._command,
                args=self._args + ["--worker_id", str(i)],
                restart_policy=restart_policy,
            )

    def remove_workers(self):
        for i in range(self._num_worker):
            pod_name = self._k8s_client.get_pod_name(i)
            with self._lock:
                if pod_name in self._pods_phase:
                    self._logger.info("Deleting worker: %d", i)
                    self._k8s_client.delete_worker(i)

    def get_counters(self):
        with self._lock:
            return Counter(self._pods_phase.values())

    def _event_cb(self, event):
        evt_obj = event.get("object")
        evt_type = event.get("type")
        if not evt_obj or not evt_type:
            self._logger.error("Event doesn't have object or type: %s" % event)
            return
        pod_name = evt_obj.metadata.name
        with self._lock:
            self._pods_phase[pod_name] = evt_obj.status.phase
            if evt_type == "DELETED":
                del self._pods_phase[pod_name]
                self._task_q.recover_tasks(
                    # TODO: move worker_id and pod name mapping to a separate
                    # class
                    int(pod_name.rsplit("-", 1)[1])
                )
