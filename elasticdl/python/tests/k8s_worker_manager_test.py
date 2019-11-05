import os
import random
import time
import unittest
from unittest.mock import MagicMock, call

from elasticdl.python.master.k8s_worker_manager import WorkerManager
from elasticdl.python.master.task_dispatcher import _TaskDispatcher


class WorkerManagerTest(unittest.TestCase):
    @unittest.skipIf(
        os.environ.get("K8S_TESTS", "True") == "False",
        "No Kubernetes cluster available",
    )
    def testCreateDeleteWorkerPod(self):
        task_d = _TaskDispatcher({"f": (0, 10)}, {}, {}, 1, 1)
        task_d.recover_tasks = MagicMock()
        worker_manager = WorkerManager(
            task_d,
            job_name="test-create-worker-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="gcr.io/google-samples/hello-app:1.0",
            worker_command=["echo"],
            worker_args=[],
            namespace="default",
            num_workers=3,
        )

        worker_manager.start_workers()
        max_check_num = 20
        for _ in range(max_check_num):
            time.sleep(3)
            counters = worker_manager.get_counters()
            print(counters)
            if counters["Succeeded"] == 3:
                break

        worker_manager.stop_relaunch_and_remove_workers()
        for _ in range(max_check_num):
            time.sleep(3)
            counters = worker_manager.get_counters()
            print(counters)
            if not counters:
                break
        task_d.recover_tasks.assert_has_calls(
            [call(0), call(1), call(2)], any_order=True
        )

    @unittest.skipIf(
        os.environ.get("K8S_TESTS", "True") == "False",
        "No Kubernetes cluster available",
    )
    def testFailedWorkerPod(self):
        """
        Start a pod running a python program destined to fail with
        restart_policy="Never" to test failed_worker_count
        """
        task_d = _TaskDispatcher({"f": (0, 10)}, {}, {}, 1, 1)
        task_d.recover_tasks = MagicMock()
        worker_manager = WorkerManager(
            task_d,
            job_name="test-failed-worker-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="gcr.io/google-samples/hello-app:1.0",
            worker_command=["badcommand"],
            worker_args=["badargs"],
            namespace="default",
            num_workers=3,
            restart_policy="Never",
        )
        worker_manager.start_workers()
        max_check_num = 20
        for _ in range(max_check_num):
            time.sleep(3)
            counters = worker_manager.get_counters()
            print(counters)
            if counters["Failed"] == 3:
                break

        worker_manager.stop_relaunch_and_remove_workers()
        for _ in range(max_check_num):
            time.sleep(3)
            counters = worker_manager.get_counters()
            print(counters)
            if not counters:
                break
        task_d.recover_tasks.assert_has_calls(
            [call(0), call(1), call(2)], any_order=True
        )

    @unittest.skipIf(
        os.environ.get("K8S_TESTS", "True") == "False",
        "No Kubernetes cluster available",
    )
    def testRelaunchWorkerPod(self):
        task_d = _TaskDispatcher({"f": (0, 10)}, {}, {}, 1, 1)
        worker_manager = WorkerManager(
            task_d,
            job_name="test-relaunch-worker-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="gcr.io/google-samples/hello-app:1.0",
            worker_command=["sleep 10"],
            worker_args=[],
            namespace="default",
            num_workers=3,
        )

        worker_manager.start_workers()

        max_check_num = 60
        for _ in range(max_check_num):
            time.sleep(1)
            counters = worker_manager.get_counters()
            print(counters)
            if counters["Running"] + counters["Pending"] > 0:
                break
        # Note: There is a slight chance of race condition.
        # Hack to find a worker to remove
        current_workers = set()
        live_workers = set()
        with worker_manager._lock:
            for k, (_, phase) in worker_manager._worker_pods_phase.items():
                current_workers.add(k)
                if phase in ["Running", "Pending"]:
                    live_workers.add(k)
        self.assertTrue(live_workers)

        worker_manager._remove_worker(live_workers.pop())
        # verify a new worker get launched
        found = False
        print(current_workers)
        for _ in range(max_check_num):
            if found:
                break
            time.sleep(1)
            counters = worker_manager.get_counters()
            print(counters)
            with worker_manager._lock:
                for k in worker_manager._worker_pods_phase:
                    if k not in current_workers:
                        found = True
        else:
            self.fail("Failed to find newly launched worker.")

        worker_manager.stop_relaunch_and_remove_workers()


if __name__ == "__main__":
    unittest.main()
