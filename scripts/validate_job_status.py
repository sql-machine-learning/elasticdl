import sys
import time

from kubernetes import client, config


class Client(object):
    def __init__(self, namespace):
        self.namespace = namespace
        config.load_kube_config()
        self.client = client.CoreV1Api()

    def get_pod_phase(self, pod_name):
        try:
            pod = self.client.read_namespaced_pod(
                namespace=self.namespace, name=pod_name
            )
            return pod.status.phase
        except Exception:
            return "Pod %s not found" % pod_name

    def get_pod_status(self, pod_name):
        try:
            pod = self.client.read_namespaced_pod(
                namespace=self.namespace, name=pod_name
            )
            return pod.status.to_str()
        except Exception:
            return "Pod %s not found" % pod_name

    def get_pod_label_status(self, pod_name):
        try:
            pod = self.client.read_namespaced_pod(
                namespace=self.namespace, name=pod_name
            )
            return pod.metadata.labels["status"]
        except Exception:
            return "Pod %s not found" % pod_name

    def get_pod_log(self, pod_name):
        try:
            return self.client.read_namespaced_pod_log(
                namespace=self.namespace, name=pod_name
            )
        except Exception:
            return "Pod %s not found" % pod_name

    def delete_pod(self, pod_name):
        self.client.delete_namespaced_pod(
            pod_name,
            self.namespace,
            body=client.V1DeleteOptions(grace_period_seconds=0),
        )


def check_success(statuses):
    for status in statuses:
        if status != "Succeeded":
            return False
    return True


def check_failed(statuses):
    for status in statuses:
        if status == "Failed":
            return True
    return False


def validate_job_status(client, job_type, ps_num, worker_num):
    ps_pod_names = [
        "elasticdl-test-" + job_type + "-ps-" + str(i) for i in range(ps_num)
    ]
    worker_pod_names = [
        "elasticdl-test-" + job_type + "-worker-" + str(i)
        for i in range(worker_num)
    ]
    master_pod_name = "elasticdl-test-" + job_type + "-master"

    for step in range(200):
        master_pod_phase = client.get_pod_phase(master_pod_name)
        ps_pod_phases = [client.get_pod_phase(ps) for ps in ps_pod_names]
        worker_pod_phases = [
            client.get_pod_phase(worker) for worker in worker_pod_names
        ]

        if (
            check_success([master_pod_phase])
            and check_success(ps_pod_phases)
            and check_success(worker_pod_phases)
        ):
            print("ElasticDL job succeeded.")
            client.delete_pod(master_pod_name)
            return
        elif (
            check_success(ps_pod_phases)
            and check_success(worker_pod_phases)
            and client.get_pod_phase(master_pod_name) == "Running"
            and client.get_pod_label_status(master_pod_name) == "Finished"
        ):
            print(
                "ElasticDL job succeeded"
                "(master pod keeps running for TensorBoard service)."
            )
            client.delete_pod(master_pod_name)
            return
        elif (
            check_failed(ps_pod_phases)
            or check_failed(worker_pod_phases)
            or check_failed([master_pod_phase])
        ):
            print("ElasticDL job failed.")
            print(client.get_pod_status(master_pod_name))
            print("Master log:")
            print(client.get_pod_log(master_pod_name))
            for i, ps in enumerate(ps_pod_names):
                print("PS%d log" % i)
                print(client.get_pod_log(ps))
            for i, worker in enumerate(worker_pod_names):
                print("Worker%d log" % i)
                print(client.get_pod_log(worker))
            client.delete_pod(master_pod_name)
            return
        else:
            print("Master: %s" % client.get_pod_phase(master_pod_name))
            for i, ps in enumerate(ps_pod_names):
                print("PS%d: %s" % (i, client.get_pod_phase(ps)))
            for i, worker in enumerate(worker_pod_names):
                print("Worker%d: %s" % (i, client.get_pod_phase(worker)))
                time.sleep(10)

    print("ElasticDL job timed out.")
    client.delete_pod(master_pod_name)


if __name__ == "__main__":
    k8s_client = Client(namespace="default")
    job_type = sys.argv[1]
    ps_num = int(sys.argv[2])
    worker_num = int(sys.argv[3])
    validate_job_status(k8s_client, job_type, ps_num, worker_num)
