import logging

from collections import Counter
from elasticdl.python.elasticdl.master import k8s


class _WorkerTracker(object):
    def __init__(self, task_q):
        # pod name to phase mapping
        # phase: Pending/Running/Succeeded/Failed/Unknown
        #   Pending: worker pod not started yet
        #   Running: worker pod is running
        #   Succeeded: worker pod finishes all jobs and terminates with no issue.
        #   Failed: worker pod is killed for some reason
        #   Unknown: unkown
        self._pods_phase = {}
        self._task_q = task_q

    def get_counters(self):
        return Counter(self._pods_phase.values())

    def event_cb(self, event):
        pod_name = event["object"].metadata.name
        self._pods_phase[pod_name] = event["object"].status.phase
        if event["type"] == "DELETED":
            del self._pods_phase[pod_name]
            self._task_q.recover_tasks(
                # TODO: move worker_id and pod name mapping to a separate class 
                int(pod_name.rsplit("-", 1)[1])
            )


class WorkerManager(object):
    def __init__(self, task_q, command, args, num_worker=1, cpu_request="1000m", cpu_limit="1000m",
                 memory_request="4096Mi", memory_limit="4096Mi", pod_priority=None,
                 mount_path=None, volume_name=None, **kwargs):
        self._logger = logging.getLogger("WorkerManager")
        self._command = command
        self._args = args
        self._num_worker = num_worker
        self._cpu_request = cpu_request 
        self._cpu_limit = cpu_limit
        self._memory_request = memory_request
        self._memory_limit = memory_limit
        self._pod_priority = pod_priority
        self._mount_path = mount_path
        self._volume_name = volume_name
        self._worker_tracker = _WorkerTracker(task_q)
        self._k8s_client = k8s.Client(
            event_callback=self._worker_tracker.event_cb, **kwargs
        )

    def start_workers(self, restart_policy="OnFailure"):
        for i in range(self._num_worker):
            self._logger.info("Starting worker: %d" % i)
            self._add_worker(i, restart_policy=restart_policy)

    def remove_workers(self):
        for i in range(self._num_worker):
            pod_name = self._k8s_client.get_pod_name(i)
            if pod_name in self._worker_tracker._pods_phase:
                self._logger.info("Deleting worker: %d", i)
                self._delete_worker(i)

    def _add_worker(self, worker_id, restart_policy):
        self._k8s_client.create_worker(
            worker_id,
            self._cpu_request,
            self._cpu_limit,
            self._memory_request,
            self._memory_limit,
            self._pod_priority,
            self._mount_path,
            self._volume_name,
            command=self._command,
            args=self._args + ["--worker_id", str(worker_id)],
            restart_policy=restart_policy,
        )

    def _delete_worker(self, worker_id):
        self._k8s_client.delete_worker(worker_id)

    def get_counters(self):
        return self._worker_tracker.get_counters()
