import time
import traceback

from kubernetes import client, config

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
        self.core_api = client.CoreV1Api()

        try:
            if use_kube_config:
                config.load_kube_config()
                logger.info("Load the incluster config.")
            else:
                config.load_incluster_config()
                logger.info("Load the kube config file.")
        except Exception as ex:
            traceback.print_exc()
            raise Exception(
                "Failed to load configuration for Kubernetes:\n%s" % str(ex)
            )

    def monitor_status(self):
        retry_num = 0
        pod_succeed = False
        while True:
            pod = self.get_pod()
            if pod is None:
                time.sleep(10)
                retry_num += 1
                if retry_num > MAX_READ_POD_RETRIES:
                    logger.error("{} Not Found".format(self.pod_name))

            if pod.status.phase == PodStatus.SUCCEEDED:
                pod_succeed = True
                break
            elif pod.status.phase == PodStatus.FAILED:
                logger.info(self.get_pod_log())
                break
            else:
                logger.info(pod.status.phase)
                time.sleep(30)

        self.core_api.delete_namespaced_pod(
            name=self.pod_name,
            namespace=self.namespace,
            body=client.V1DeleteOptions(grace_period_seconds=0),
        )
        logger.info("End")
        return pod_succeed

    def get_pod(self):
        try:
            pod = self.core_api.read_namespaced_pod(
                namespace=self.namespace, name=self.pod_name
            )
            return pod
        except client.api_client.ApiException as e:
            logger.warning("Exception when reading pod: %s\n" % e)
            return None

    def get_pod_log(self):
        try:
            logs = self.core_api.read_namespaced_pod_log(
                namespace=self.namespace, name=self.pod_name
            )
            return logs
        except client.api_client.ApiException as e:
            logger.warning("Exception when reading log: %s\n" % e)
            return None


class EdlJobMonitor:
    def __init__(self, namespace, job_name, worker_num, ps_num):
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
        """
        self.worker_num = worker_num
        self.ps_num = ps_num
        self.job_name = job_name
        self.client = Client(
            image_name=None,
            namespace=namespace,
            job_name=job_name,
            force_use_kube_config_file=True,
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

    def show_master_increment_log(self, old_log):
        new_log = self.client.get_master_log()
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
        master_log = ""
        while True:
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

            logger.info("Master status: {}".format(master_pod.status.phase))
            if master_pod.status.phase == PodStatus.SUCCEEDED:
                job_succeed = True
                break
            elif master_pod.status.phase == PodStatus.PENDING:
                time.sleep(60)
            elif master_pod.status.phase == PodStatus.FAILED:
                log = self.client.get_master_log()
                print_tail_log(log, tail_num=100)
                logger.error("Job {} Failed".format(self.job_name))
                break
            else:
                master_log = self.show_master_increment_log(master_log)
                self.check_worker_status()
                self.check_ps_status()
                time.sleep(60)

        if job_succeed:
            self.client.delete_master()
        return job_succeed
