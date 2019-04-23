import os
import time

import k8s


class WorkerPodStatus(object):
    """
    phase: Pending/Running/Succeeded/Failed/Unknown
        Pending: worker pod not started yet
        Running: worker pod is running
        Succeeded: worker pod finishes all jobs and terminates with no issue.
        Failed: worker pod is killed for some reason
        Unknown: unkown
    last_event_type: ADDED/MODIFIED/DELETED
    """

    def __init__(self, phase=None, last_event_type=None):
        self.phase = phase
        self.last_event_type = last_event_type


class WorkerTracker(object):
    def __init__(self):
        self._pod_count = 0
        self._unfinished_worker_count = 0
        self._finished_worker_count = 0
        self._failed_worker_count = 0
        self._worker_pods_status = {}

    def get_counters(self):
        return {
            "pod_count": self._pod_count,
            "unfinished_worker_count": self._unfinished_worker_count,
            "finished_worker_count": self._finished_worker_count,
            "failed_worker_count": self._failed_worker_count
        }

    def update_worker_counter(self, old_phase, new_phase, event_type):
        if old_phase is None:
            self._unfinished_worker_count += 1
        if old_phase != new_phase and new_phase == "Succeeded":
            self._finished_worker_count += 1
            self._unfinished_worker_count -= 1
        if old_phase != new_phase and new_phase == "Failed":
            self._failed_worker_count += 1
            self._unfinished_worker_count -= 1

        if event_type == "ADDED":
            self._pod_count += 1
        elif event_type == "DELETED":
            self._pod_count -= 1

    def event_cb(self, event):
        container_name = event["object"].spec.containers[0].name
        if container_name not in self._worker_pods_status:
            self.update_worker_counter(None,
                                       event["object"].status.phase,
                                       event["type"])
            self._worker_pods_status[container_name] = WorkerPodStatus(
                phase=event["object"].status.phase,
                last_event_type=event["type"])
        else:
            self.update_worker_counter(
                self._worker_pods_status[container_name].phase,
                event["object"].status.phase,
                event["type"])
            self._worker_pods_status[container_name].phase = event["object"].status.phase
            self._worker_pods_status[container_name].last_event_type = event["type"]


class WorkerServicer(object):
    def __init__(self,
                 job_name,
                 worker_image,
                 command=None,
                 args=None,
                 namespace="default",
                 worker_num=1
                 ):
        self._job_name = job_name
        self._worker_image = worker_image
        self._command = command
        self._args = args
        self._namespace = namespace
        self._worker_num = worker_num

        self._worker_tracker = WorkerTracker()
        self._k8s_client = k8s.Client(
            worker_image=self._worker_image,
            namespace=self._namespace,
            job_name=self._job_name,
            master_addr="",
            event_callback=self._worker_tracker.event_cb,
        )

    def start_workers(self, restart_policy="OnFailure"):
        for i in range(self._worker_num):
            self.add_worker("worker-%d" % i, restart_policy=restart_policy)

    def remove_workers(self):
        for i in range(self._worker_num):
            worker_name = "worker-%d" % i
            for container_name in self._worker_tracker._worker_pods_status:
                if worker_name in container_name and \
                self._worker_tracker._worker_pods_status[container_name].last_event_type != "DELETED":
                    self.delete_worker(worker_name)
                    break

    def add_worker(self, worker_name, restart_policy="OnFailure"):
        self._k8s_client.create_worker(
            worker_name,
            command=self._command,
            args=self._args,
            restart_policy=restart_policy)

    def delete_worker(self, worker_name):
        self._k8s_client.delete_worker(worker_name)

    def get_counters(self):
        return self._worker_tracker.get_counters()
