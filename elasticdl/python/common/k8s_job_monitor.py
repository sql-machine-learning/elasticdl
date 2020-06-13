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

import time

from kubernetes import client

from elasticdl.python.common.constants import PodStatus
from elasticdl.python.common.k8s_client import Client
from elasticdl.python.common.log_utils import default_logger as logger

MAX_READ_POD_RETRIES = 6


def print_tail_log(log, tail_num):
    if log is not None:
        log_lines = log.split("\n")
        tail_index = -1 * tail_num
        logger.info("\n".join(log_lines[tail_index:]))


class PodMonitor:
    def __init__(self, namespace, pod_name, use_kube_config=True):
        """
        k8s Pod Monitor. During data transformation, we may launch a pod
        on k8s cluster to perform data analysis and need to monitor the
        pod execution status.

        Args:
            namespace: The name of the Kubernetes namespace where the pod
                will be created.
            pod_name: Pod name, should be unique in the namespace.
            use_kube_config: If true, load the cluster config from
                ~/.kube/config. Otherwise, if it's in a process running in
                a K8S environment, it loads the incluster config.
        """
        self.namespace = namespace
        self.pod_name = pod_name
        self.client = Client(
            image_name=None,
            namespace=namespace,
            job_name=None,
            force_use_kube_config_file=use_kube_config,
        )

    def monitor_status(self):
        retry_num = 0
        pod_succeeded = False

        while True:
            try:
                pod = self.client.get_pod(self.pod_name)
                if pod is None:
                    retry_num += 1
                    if retry_num > MAX_READ_POD_RETRIES:
                        logger.error("{} Not Found".format(self.pod_name))
                        break
                    time.sleep(10)
                    continue

                retry_num = 0

                logger.info("Pod Status : %s" % pod.status.phase)
                if pod.status.phase == PodStatus.SUCCEEDED:
                    pod_succeeded = True
                    break
                elif pod.status.phase == PodStatus.FAILED:
                    logger.info(self.client.get_pod_log(self.pod_name))
                    break
                else:
                    time.sleep(30)
            except client.api_client.ApiException:
                time.sleep(60)
        return pod_succeeded

    def delete_pod(self):
        if self.client.get_pod(self.pod_name):
            self.client.delete_pod(self.pod_name)
        # Wait until the pod is deleted
        while self.client.get_pod(self.pod_name):
            time.sleep(5)


class EdlJobMonitor:
    def __init__(
        self, namespace, job_name, worker_num, ps_num, use_kube_config=True
    ):
        """
        ElasticDL job monitor. After launching an ElasticDL job, the user
        may want to monitor the job status.

        Args:
            namespace: The name of the Kubernetes namespace where the pod
                will be created.
            job_name: ElasticDL job name, should be unique in the namespace.
                Used as pod name prefix and value for "elasticdl" label.
            worker_num: Integer, worker number of the job.
            ps_num: Integer, parameter server number of the job.
            use_kube_config: If true, load the cluster config from
                ~/.kube/config. Otherwise, if it's in a process running in
                a K8S environment, it loads the incluster config.
        """
        self.worker_num = worker_num
        self.ps_num = ps_num
        self.job_name = job_name
        self.client = Client(
            image_name=None,
            namespace=namespace,
            job_name=job_name,
            force_use_kube_config_file=use_kube_config,
        )

    def check_worker_status(self):
        for i in range(self.worker_num):
            worker_pod = self.client.get_worker_pod(i)
            worker_pod_name = self.client.get_worker_pod_name(i)
            if worker_pod is None:
                logger.error("Worker {} Not Found".format(worker_pod_name))
            elif worker_pod.status.phase == PodStatus.FAILED:
                logger.error(
                    "Worker {} {}".format(
                        worker_pod_name, worker_pod.status.phase
                    )
                )

    def check_ps_status(self):
        for i in range(self.ps_num):
            ps_pod = self.client.get_ps_pod(i)
            ps_pod_name = self.client.get_ps_pod_name(i)
            if ps_pod is None:
                logger.error("PS {} Not Found".format(ps_pod_name))
            elif ps_pod.status.phase == PodStatus.FAILED:
                logger.error(
                    "PS {} {}".format(ps_pod_name, ps_pod.status.phase)
                )

    def show_evaluation_and_task_log(self, new_log, old_log):
        """Show the master's incremental logs about evaluation task and
        latest completed task compared with the last query.
        """
        if new_log is None:
            return
        increment_log = new_log.replace(old_log, "")
        task_log = ""
        for log_line in increment_log.split("\n"):
            if "Evaluation" in log_line:
                logger.info(log_line)
            if "Task" in log_line:
                task_log = log_line
        logger.info(task_log)
        return new_log

    def monitor_status(self):
        retry_num = 0
        job_succeed = False
        master_old_log = ""
        while True:
            try:
                master_pod = self.client.get_master_pod()
                if master_pod is None:
                    retry_num += 1
                    if retry_num > MAX_READ_POD_RETRIES:
                        logger.error(
                            "{} Not Found".format(
                                self.client.get_master_pod_name()
                            )
                        )
                        break
                    time.sleep(10)
                    continue
                retry_num = 0

                logger.info(
                    "Master status: {}".format(master_pod.status.phase)
                )
                if master_pod.status.phase == PodStatus.SUCCEEDED:
                    job_succeed = True
                    break
                elif master_pod.status.phase == PodStatus.PENDING:
                    time.sleep(10)
                elif master_pod.status.phase == PodStatus.FAILED:
                    log = self.client.get_master_log()
                    print_tail_log(log, tail_num=100)
                    logger.error("Job {} Failed".format(self.job_name))
                    break
                else:
                    master_new_log = self.client.get_master_log()
                    self.show_evaluation_and_task_log(
                        master_new_log, master_old_log
                    )
                    master_old_log = master_new_log
                    time.sleep(60)
            except client.api_client.ApiException:
                time.sleep(60)
        return job_succeed

    def delete_job(self):
        if self.client.get_master_pod():
            self.client.delete_master()

        # Wait until the master is deleted
        while self.client.get_master_pod():
            time.sleep(5)
