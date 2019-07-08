import os

from elasticdl.python.common import k8s_client as k8s
from elasticdl.python.elasticdl.image_builder import (
    build_and_push_docker_image,
)

MODEL_ROOT_PATH = "/model_zoo"


def train(args):
    image_name = build_and_push_docker_image(
        model_zoo=args.model_def,
        base_image=args.image_base,
        docker_image_prefix=args.docker_image_prefix,
        extra_pypi=args.extra_pypi_index,
    )
    container_args = [
        "-m",
        "elasticdl.python.master.main",
        "--job_name",
        args.job_name,
        "--worker_image",
        image_name,
        "--model_def",
        _model_def_in_docker(args.model_def),
        "--num_workers",
        str(args.num_workers),
        "--worker_resource_request",
        args.worker_resource_request,
        "--worker_resource_limit",
        args.worker_resource_limit,
        "--namespace",
        args.namespace,
        "--tensorboard_log_dir",
        args.tensorboard_log_dir,
        "--records_per_task",
        str(args.records_per_task),
        "--num_epochs",
        str(args.num_epochs),
        "--grads_to_wait",
        str(args.grads_to_wait),
        "--minibatch_size",
        str(args.minibatch_size),
        "--training_data_dir",
        args.training_data_dir,
        "--evaluation_data_dir",
        args.evaluation_data_dir,
        "--evaluation_steps",
        str(args.evaluation_steps),
        "--evaluation_start_delay_secs",
        str(args.evaluation_start_delay_secs),
        "--evaluation_throttle_secs",
        str(args.evaluation_throttle_secs),
    ]
    container_args.extend(["--image_pull_policy", args.image_pull_policy])
    container_args.extend(["--restart_policy", args.restart_policy])

    if all([args.volume_name, args.mount_path]):
        container_args.extend(
            [
                "--mount_path",
                args.mount_path,
                "--volume_name",
                args.volume_name,
            ]
        )
    elif any([args.volume_name, args.mount_path]):
        raise ValueError(
            "Not both of the parameters volume_name and "
            "mount_path are provided."
        )

    k8s.Client(
        image_name=image_name,
        namespace=args.namespace,
        job_name=args.job_name,
        event_callback=None,
    ).create_master(
        resource_requests=args.master_resource_request,
        resource_limits=args.master_resource_limit,
        args=container_args,
        pod_priority=args.master_pod_priority,
        image_pull_policy=args.image_pull_policy,
        restart_policy=args.restart_policy,
        volume_name=args.volume_name,
        mount_path=args.mount_path,
    )
    # TODO: print dashboard url after launching the master pod


def evaluate(args):
    image_name = build_and_push_docker_image(
        model_zoo=args.model_def,
        base_image=args.image_base,
        docker_image_prefix=args.docker_image_prefix,
        extra_pypi=args.extra_pypi_index,
    )
    container_args = [
        "-m",
        "elasticdl.python.master.main",
        "--job_name",
        args.job_name,
        "--worker_image",
        image_name,
        "--model_def",
        _model_def_in_docker(args.model_def),
        "--num_workers",
        str(args.num_workers),
        "--worker_resource_request",
        args.worker_resource_request,
        "--worker_resource_limit",
        args.worker_resource_limit,
        "--namespace",
        args.namespace,
        "--records_per_task",
        str(args.records_per_task),
        "--minibatch_size",
        str(args.minibatch_size),
        "--evaluation_data_dir",
        args.evaluation_data_dir,
        "--checkpoint_filename_for_init",
        args.checkpoint_filename_for_init,
    ]
    container_args.extend(["--image_pull_policy", args.image_pull_policy])
    container_args.extend(["--restart_policy", args.restart_policy])

    if all([args.volume_name, args.mount_path]):
        container_args.extend(
            [
                "--mount_path",
                args.mount_path,
                "--volume_name",
                args.volume_name,
            ]
        )
    elif any([args.volume_name, args.mount_path]):
        raise ValueError(
            "Not both of the parameters volume_name and "
            "mount_path are provided."
        )

    k8s.Client(
        image_name=image_name,
        namespace=args.namespace,
        job_name=args.job_name,
        event_callback=None,
    ).create_master(
        resource_requests=args.master_resource_request,
        resource_limits=args.master_resource_limit,
        args=container_args,
        pod_priority=args.master_pod_priority,
        image_pull_policy=args.image_pull_policy,
        restart_policy=args.restart_policy,
        volume_name=args.volume_name,
        mount_path=args.mount_path,
    )


def _model_def_in_docker(model_def):
    return os.path.join(MODEL_ROOT_PATH, os.path.basename(model_def))
