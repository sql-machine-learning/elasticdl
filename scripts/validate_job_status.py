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

import sys
import time

from kubernetes import client, config


def print_tail_log(log, tail_num):
    if log is not None:
        log_lines = log.split("\n")
        tail_index = -1 * tail_num
        print("\n".join(log_lines[tail_index:]))


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

    for step in range(10):
        print("Query master pod phase")
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
            exit(0)
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
            exit(0)
        elif (
            check_failed(ps_pod_phases)
            or check_failed(worker_pod_phases)
            or check_failed([master_pod_phase])
        ):
            print("ElasticDL job failed.")
            print(client.get_pod_status(master_pod_name))
            print("Master log:")
            print(client.get_pod_log(master_pod_name))
            for ps, pod_phase in zip(ps_pod_names, ps_pod_phases):
                if check_failed([pod_phase]):
                    print("PS %s log" % ps)
                    print_tail_log(client.get_pod_log(ps), 50)
            for worker, pod_phase in zip(worker_pod_names, worker_pod_phases):
                if check_failed([pod_phase]):
                    print("Worker %s log" % worker)
                    print_tail_log(client.get_pod_log(worker), 50)
            client.delete_pod(master_pod_name)
            exit(-1)
        else:
            print(
                "Master (status.phase): %s"
                % client.get_pod_phase(master_pod_name)
            )
            print(
                "Master (metadata.labels.status): %s"
                % client.get_pod_label_status(master_pod_name)
            )
            for i, ps in enumerate(ps_pod_names):
                print("PS%d: %s" % (i, client.get_pod_phase(ps)))
            for i, worker in enumerate(worker_pod_names):
                print("Worker%d: %s" % (i, client.get_pod_phase(worker)))
            time.sleep(10)

    print("ElasticDL job timed out.")
    client.delete_pod(master_pod_name)
    exit(-1)


if __name__ == "__main__":
    print("Start validate job status")
    k8s_client = Client(namespace="default")
    job_type = sys.argv[1]
    ps_num = int(sys.argv[2])
    worker_num = int(sys.argv[3])
    print("Job args :{}".format(job_type))
    validate_job_status(k8s_client, job_type, ps_num, worker_num)
