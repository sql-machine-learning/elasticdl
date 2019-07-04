import os

from elasticdl.python.common import k8s_client as k8s
from elasticdl.python.elasticdl.image_builder import (
    build_and_push_docker_image,
)

MODEL_ROOT_PATH = "/model_zoo"


def train(
    job_name,
    namespace,
    model_def,
    master_resource_request,
    master_resource_limit,
    num_workers,
    worker_resource_request,
    worker_resource_limit,
    master_pod_priority,
    image_base,
    docker_image_prefix,
    extra_pypi_index,
    tensorboard_log_dir,
    image_pull_policy,
    restart_policy,
    volume_name,
    mount_path,
    records_per_task,
    num_epochs,
    grads_to_wait,
    minibatch_size,
    training_data_dir,
    evaluation_data_dir,
):
    image_name = build_and_push_docker_image(
        model_zoo=model_def,
        base_image=image_base,
        docker_image_prefix=docker_image_prefix,
        extra_pypi=extra_pypi_index,
    )
    container_args = [
        "-m",
        "elasticdl.python.master.main",
        "--job_name",
        job_name,
        "--worker_image",
        image_name,
        "--model_def",
        _model_def_in_docker(model_def),
        "--num_workers",
        str(num_workers),
        "--worker_resource_request",
        worker_resource_request,
        "--worker_resource_limit",
        worker_resource_limit,
        "--namespace",
        namespace,
        "--tensorboard_log_dir",
        tensorboard_log_dir,
        "--records_per_task",
        str(records_per_task),
        "--num_epochs",
        str(num_epochs),
        "--grads_to_wait",
        str(grads_to_wait),
        "--minibatch_size",
        str(minibatch_size),
        "--training_data_dir",
        training_data_dir,
        "--evaluation_data_dir",
        evaluation_data_dir,
    ]
    container_args.extend(["--image_pull_policy", image_pull_policy])
    container_args.extend(["--restart_policy", restart_policy])

    if all([volume_name, mount_path]):
        container_args.extend(
            ["--mount_path", mount_path, "--volume_name", volume_name]
        )
    elif any([volume_name, mount_path]):
        raise ValueError(
            "Not both of the parameters volume_name and "
            "mount_path are provided."
        )

    k8s.Client(
        image_name=image_name,
        namespace=namespace,
        job_name=job_name,
        event_callback=None,
    ).create_master(
        resource_requests=master_resource_request,
        resource_limits=master_resource_limit,
        args=container_args,
        pod_priority=master_pod_priority,
        image_pull_policy=image_pull_policy,
        restart_policy=restart_policy,
        volume_name=volume_name,
        mount_path=mount_path,
    )
    # TODO: print dashboard url after launching the master pod


def evaluate(args, argv):
    # TODO implement distributed evaluation.
    raise NotImplementedError()


def _model_def_in_docker(model_def):
    return os.path.join(MODEL_ROOT_PATH, os.path.basename(model_def))
