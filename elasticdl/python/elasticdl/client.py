import argparse

from elasticdl.python.elasticdl.api import evaluate, train


def main():
    parser = argparse.ArgumentParser(
        usage="""elasticdl <command> [<args>]

Below is the list of supported commands:
train         Submit a ElasticDL distributed training job.
evaluate      Submit a ElasticDL distributed evaluation job.
"""
    )
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train", help="elasticdl train -h")
    train_parser.set_defaults(func=train)
    _add_train_params(train_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="elasticdl evaluate -h"
    )
    evaluate_parser.set_defaults(func=evaluate)
    _add_evaluate_params(evaluate_parser)

    args, argv = parser.parse_known_args()
    args.func(args)


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
        "--volume",
        help="The Kubernetes volume information, "
        'e.g. "claim_name=c1,mount_path=/path1".',
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
    parser.add_argument(
        "--training_data_dir",
        help="Training data directory. Files should be in RecordIO format",
        required=True,
    )
    parser.add_argument(
        "--evaluation_data_dir",
        help="Evaluation data directory. Files should be in RecordIO format",
        default="",
    )
    parser.add_argument(
        "--evaluation_steps",
        type=int,
        help="Evaluate the model every this many steps."
        "If 0, step-based evaluation is disabled",
        default=0,
    )
    parser.add_argument(
        "--evaluation_start_delay_secs",
        type=int,
        help="Start time-based evaluation only after waiting for "
        "this many seconds",
        default=100,
    )
    parser.add_argument(
        "--evaluation_throttle_secs",
        type=int,
        help="Do not re-evaluate unless the last evaluation was started "
        "at least this many seconds ago."
        "If 0, time-based evaluation is disabled",
        default=0,
    )
    parser.add_argument(
        "--checkpoint_steps",
        type=int,
        help="Save checkpoint every this many steps."
        "If 0, no checkpoints to save.",
        default=0,
    )
    parser.add_argument(
        "--checkpoint_dir",
        help="The directory to store the checkpoint files",
        default="",
    )
    parser.add_argument(
        "--keep_checkpoint_max",
        type=int,
        help="The maximum number of recent checkpoint files to keep."
        "If 0, keep all.",
        default=0,
    )
    parser.add_argument(
        "--checkpoint_filename_for_init",
        help="The checkpoint file to initialize the training model",
        default="",
    )
    parser.add_argument(
        "--input_fn",
        type=str,
        default="input_fn",
        help="The name of the input function defined in the model file",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="loss",
        help="The name of the loss function defined in the model file",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="optimizer",
        help="The name of the optimizer defined in the model file",
    )
    parser.add_argument(
        "--eval_metrics_fn",
        type=str,
        default="eval_metrics_fn",
        help="The name of the evaluation metrics function defined "
        "in the model file",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="model",
        help="The name of the model class defined in the model file",
    )
    parser.add_argument(
        "--model_params",
        type=str,
        default="",
        help="The dictionary of model parameters in a string that will be "
        'used to instantiate the model, e.g. "param1=1,param2=2"',
    )
    parser.add_argument(
        "--cluster_spec",
        help="The file that contains user-defined cluster specification",
        default="",
    )


def _add_evaluate_params(parser):
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
        "--volume",
        help="The Kubernetes volume information, "
        'e.g. "claim_name=c1,mount_path=/path1".',
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
    parser.add_argument("--records_per_task", type=int, required=True)
    parser.add_argument(
        "--minibatch_size",
        type=int,
        help="Minibatch size used by workers",
        required=True,
    )
    parser.add_argument(
        "--evaluation_data_dir",
        help="Evaluation data directory. Files should be in RecordIO format",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_filename_for_init",
        help="The checkpoint file to initialize the training model",
        required=True,
    )
    parser.add_argument(
        "--input_fn",
        type=str,
        default="input_fn",
        help="The name of the input function defined in the model file",
    )
    parser.add_argument(
        "--eval_metrics_fn",
        type=str,
        default="eval_metrics_fn",
        help="The name of the evaluation metrics function defined "
        "in the model file",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="model",
        help="The name of the model class defined in the model file",
    )
    parser.add_argument(
        "--model_params",
        type=str,
        default="",
        help="The dictionary of model parameters in a string that will be "
        'used to instantiate the model, e.g. "param1=1,param2=2"',
    )
    parser.add_argument(
        "--cluster_spec",
        help="The file that contains user-defined cluster specification",
        default="",
    )


if __name__ == "__main__":
    main()
