import os
import random
import time
import unittest

from elasticdl.python.master.k8s_instance_manager import InstanceManager


class InstanceManagerTest(unittest.TestCase):
    @unittest.skipIf(
        os.environ.get("K8S_TESTS", "True") == "False",
        "No Kubernetes cluster available",
    )
    def testRelaunchPsPod(self):
        num_ps = 3
        instance_manager = InstanceManager(
            task_d=None,
            job_name="test-relaunch-ps-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="gcr.io/google-samples/hello-app:1.0",
            ps_command=["sleep 10"],
            ps_args=[],
            namespace="default",
            num_ps=num_ps,
        )

        instance_manager.start_parameter_servers()

        # Check we also have ps services started
        for i in range(num_ps):
            service = instance_manager._k8s_client.get_ps_service(i)
            self.assertTrue(service.metadata.owner_references)
            owner = service.metadata.owner_references[0]
            self.assertEqual(owner.kind, "Pod")
            self.assertEqual(
                owner.name, instance_manager._k8s_client.get_ps_pod_name(i)
            )

        max_check_num = 60
        for _ in range(max_check_num):
            time.sleep(1)
            counters = instance_manager.get_ps_counter()
            if counters["Running"] + counters["Pending"] > 0:
                break
        # Note: There is a slight chance of race condition.
        # Hack to find a ps to remove
        all_current_ps = set()
        all_live_ps = set()
        with instance_manager._lock:
            for _, (k, phase) in instance_manager._ps_pods_phase.items():
                all_current_ps.add(k)
                if phase in ["Running", "Pending"]:
                    all_live_ps.add(k)
        self.assertTrue(all_live_ps)

        ps_to_be_removed = all_live_ps.pop()
        all_current_ps.remove(ps_to_be_removed)
        instance_manager._remove_ps(ps_to_be_removed)
        # Verify a new ps gets launched
        found = False
        for _ in range(max_check_num):
            if found:
                break
            time.sleep(1)
            with instance_manager._lock:
                for _, (k, _) in instance_manager._ps_pods_phase.items():
                    if k not in all_current_ps:
                        found = True
        else:
            self.fail("Failed to find newly launched ps.")

        instance_manager.stop_relaunch_and_remove_all_ps()


if __name__ == "__main__":
    unittest.main()
