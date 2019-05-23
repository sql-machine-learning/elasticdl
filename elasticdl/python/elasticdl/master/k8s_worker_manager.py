import logging
import os
import time

from collections import Counter
from elasticdl.master import k8s


class WorkerTracker(object):
    def __init__(self):
        # pod name to phase mapping
        # phase: Pending/Running/Succeeded/Failed/Unknown
        #   Pending: worker pod not started yet
        #   Running: worker pod is running
        #   Succeeded: worker pod finishes all jobs and terminates with no issue.
        #   Failed: worker pod is killed for some reason
        #   Unknown: unkown
        self._pods_phase = {}

    def get_counters(self):
        return Counter(self._pods_phase.values())

    def event_cb(self, event):
        pod_name = event["object"].metadata.name
        self._pods_phase[pod_name] = event["object"].status.phase
        if event["type"] == "DELETED":
            del self._pods_phase[pod_name]


class WorkerManager(object):
    def __init__(self, command, args, num_worker=1, cpu_request="1000m", cpu_limit="1000m",
            memory_request="4096Mi", memory_limit="4096Mi", pod_priority=None, **kwargs):
        self._logger = logging.getLogger("WorkerManager")
        self._command = command
        self._args = args
        self._num_worker = num_worker
        self._cpu_request = cpu_request 
        self._cpu_limit = cpu_limit
        self._memory_request = memory_request
        self._memory_limit = memory_limit
        self._pod_priority = pod_priority
        self._worker_tracker = WorkerTracker()
        self._k8s_client = k8s.Client(
            event_callback=self._worker_tracker.event_cb, **kwargs
        )

    def start_workers(self, restart_policy="OnFailure"):
        for i in range(self._num_worker):
            worker_name = "%d" % i
            self._logger.warning("Starting worker: %d", i)
            self._add_worker(worker_name, restart_policy=restart_policy)

    def remove_workers(self):
        for i in range(self._num_worker):
            worker_name = "%d" % i
            pod_name = self._k8s_client.get_pod_name(worker_name)
            if pod_name in self._worker_tracker._pods_phase:
                self._logger.warning("Deleting worker: %d", i)
                self._delete_worker(worker_name)

    def _add_worker(self, worker_name, restart_policy):
        self._k8s_client.create_worker(
            worker_name,
            self._cpu_request,
            self._cpu_limit,
            self._memory_request,
            self._memory_limit,
            self._pod_priority,
            command=self._command,
            args=self._args,
            restart_policy=restart_policy,
        )

    def _delete_worker(self, worker_name):
        self._k8s_client.delete_worker(worker_name)

    def get_counters(self):
        return self._worker_tracker.get_counters()
