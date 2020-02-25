import sys
import time

from elasticdl.python.common.k8s_client import Client


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
    for step in range(200):
        master_pod = client.get_master_pod()
        master_pod_phase = (
            master_pod.status.phase if master_pod is not None else None
        )
        master_label_status = (
            master_pod.metadata.labels["status"]
            if master_pod is not None
            else None
        )
        ps_pod_phases = []
        for i in range(ps_num):
            ps_pod = client.get_ps_pod(i)
            ps_pod_phases.append(
                ps_pod.status.phase if ps_pod is not None else None
            )
        worker_pod_phases = []
        for i in range(worker_num):
            worker_pod = client.get_worker_pod(i)
            worker_pod_phases.append(
                worker_pod.status.phase if worker_pod is not None else None
            )

        if (
            check_success([master_pod_phase])
            and check_success(ps_pod_phases)
            and check_success(worker_pod_phases)
        ):
            print("ElasticDL job succeeded.")
            client.delete_master()
            exit(0)
        elif (
            check_success(ps_pod_phases)
            and check_success(worker_pod_phases)
            and master_pod_phase == "Running"
            and master_label_status == "Finished"
        ):
            print(
                "ElasticDL job succeeded"
                "(master pod keeps running for TensorBoard service)."
            )
            client.delete_master()
            exit(0)
        elif (
            check_failed(ps_pod_phases)
            or check_failed(worker_pod_phases)
            or check_failed([master_pod_phase])
        ):
            print("ElasticDL job failed.")
            if master_pod is not None:
                print(master_pod.status)
            else:
                print("Not Found {}".format(client.get_master_pod_name()))
            print("Master log:")
            print(client.get_master_log())
            for i in range(ps_num):
                print("PS%d log" % i)
                print(client.get_ps_log(i))
            for i, worker in range(worker_num):
                print("Worker%d log" % i)
                print(client.get_worker_log(i))
            client.delete_master()
            exit(-1)
        else:
            print(
                "Master (status.phase): %s" % master_pod_phase
            )
            print(
                "Master (metadata.labels.status): %s" % master_label_status
            )
            for i, phase in enumerate(ps_pod_phases):
                print("PS%d: %s" % (i, phase))
            for i, phase in enumerate(worker_pod_phases):
                print("Worker%d: %s" % (i, phase))
            time.sleep(10)

    print("ElasticDL job timed out.")
    client.delete_master()
    exit(-1)


if __name__ == "__main__":
    job_type = sys.argv[1]
    ps_num = int(sys.argv[2])
    worker_num = int(sys.argv[3])

    k8s_client = Client(
        image_name="",
        namespace="default",
        job_name="test-" + job_type,
        event_callback=None,
        cluster_spec="",
        force_use_kube_config_file=True
    )
    validate_job_status(k8s_client, job_type, ps_num, worker_num)
