import argparse

from elasticdl.python.elasticdl.api import train, evaluate


def main():
    parser = argparse.ArgumentParser(
        usage="""client.py <command> [<args>]

There are all the supported commands:
train         Submit a ElasticDL distributed training job.
evaluate      Submit a ElasticDL distributed evaluation job.
"""
    )
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser("train", help="elasticdl.py train -h")
    train_parser.set_defaults(func=train)
    _add_train_params(train_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="client.py evaluate -h"
    )
    evaluate_parser.set_defaults(func=evaluate)
    _add_evaluate_params(evaluate_parser)

    args, argv = train_parser.parse_known_args()
    args.func(
        args.job_name,
        args.namespace,
        args.model_def,
        args.master_resource_request,
        args.master_resource_limit,
        args.num_workers,
        args.worker_resource_request,
        args.worker_resource_limit,
        args.master_pod_priority,
        args.image_base,
        args.docker_image_prefix,
        args.extra_pypi_index,
        args.tensorboard_log_dir,
        args.image_pull_policy,
        args.restart_policy,
        args.volume_name,
        args.mount_path,
        args.records_per_task,
        args.num_epochs,
        args.grads_to_wait,
        args.minibatch_size,
    )


def _add_train_params(parser):
    parser.add_argument(
        "--model_def",
        help="The directory that contains user-defined model files "
        "or a specific model file",
        required=True,
    )
    parser.add_argument(
        "--docker_image_prefix",
        default="",
        help="The prefix for generated Docker images, if set, the image is "
        "also pushed to the registry",
    )
    parser.add_argument("--image_base", help="Base Docker image.")
    parser.add_argument("--job_name", help="ElasticDL job name", required=True)
    parser.add_argument(
        "--master_resource_request",
        default="cpu=0.1,memory=1024Mi",
        type=str,
        help="The minimal resource required by master, "
        "e.g. cpu=0.1,memory=1024Mi,disk=1024Mi,gpu=1",
    )
    parser.add_argument(
        "--master_resource_limit",
        type=str,
        help="The maximal resource required by master, "
        "e.g. cpu=0.1,memory=1024Mi,disk=1024Mi,gpu=1, "
        "default to master_resource_request",
    )
    parser.add_argument(
        "--num_workers", type=int, help="Number of workers", default=0
    )
    parser.add_argument(
        "--worker_resource_request",
        default="cpu=1,memory=4096Mi",
        type=str,
        help="The minimal resource required by worker, "
        "e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1",
    )
    parser.add_argument(
        "--worker_resource_limit",
        type=str,
        help="The maximal resource required by worker, "
        "e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1,"
        "default to worker_resource_request",
    )
    parser.add_argument(
        "--master_pod_priority", help="The requested priority of master pod"
    )
    parser.add_argument(
        "--volume_name", help="The volume name of network file system"
    )
    parser.add_argument(
        "--mount_path", help="The mount path in the Docker container"
    )
    parser.add_argument(
        "--image_pull_policy",
        default="Always",
        help="The image pull policy of master and worker",
    )
    parser.add_argument(
        "--restart_policy",
        default="Never",
        help="The pod restart policy when pod crashed",
    )
    parser.add_argument(
        "--extra_pypi_index", help="The extra python package repository"
    )
    parser.add_argument(
        "--namespace",
        default="default",
        type=str,
        help="The name of the Kubernetes namespace where ElasticDL "
        "pods will be created",
    )
    parser.add_argument(
        "--tensorboard_log_dir",
        default="",
        type=str,
        help="Directory where TensorBoard will look to find "
        "TensorFlow event files that it can display. "
        "TensorBoard will recursively walk the directory "
        "structure rooted at log dir, looking for .*tfevents.* "
        "files. You may also pass a comma separated list of log "
        "directories, and TensorBoard will watch each "
        "directory.",
    )
    parser.add_argument("--records_per_task", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument(
        "--grads_to_wait",
        type=int,
        help="Number of gradients to wait before updating model",
        required=True,
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        help="Minibatch size used by workers to compute gradients",
        required=True,
    )


def _add_evaluate_params(parser):
    # TODO add parameters for evaluation parser..
    pass


if __name__ == "__main__":
    main()
