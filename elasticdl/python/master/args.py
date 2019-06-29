import argparse


def _pos_int(arg):
    res = int(arg)
    if res <= 0:
        raise ValueError("Positive integer argument required. Got %s" % res)
    return res


def _non_neg_int(arg):
    res = int(arg)
    if res < 0:
        raise ValueError(
            "Non-negative integer argument required. Get %s" % res
        )
    return res


def parse_args():
    parser = argparse.ArgumentParser(description="ElasticDL Master")
    parser.add_argument("--port", type=_pos_int, default=50001, required=True)
    parser.add_argument(
        "--model_file",
        help="Full file path of user defined neural model",
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
        "--evaluation_start_delay_secs",
        type=_pos_int,
        help="Start evaluation only after waiting for this many seconds",
        default=100,
    )
    parser.add_argument(
        "--evaluation_throttle_secs",
        type=_pos_int,
        help="Do not re-evaluate unless the last evaluation was started "
        "at least this many seconds ago",
        default=100,
    )
    parser.add_argument("--records_per_task", type=_pos_int, required=True)
    parser.add_argument("--num_epochs", type=_pos_int, required=True)
    parser.add_argument(
        "--grads_to_wait",
        type=_pos_int,
        help="Number of gradients to wait before updating model",
        required=True,
    )
    parser.add_argument(
        "--minibatch_size",
        type=_pos_int,
        help="Minibatch size used by workers to compute gradients",
        required=True,
    )
    parser.add_argument(
        "--num_workers", type=_pos_int, help="Number of workers", default=0
    )
    parser.add_argument(
        "--checkpoint_filename_for_init",
        help="The checkpoint file to initialize the training model",
        default="",
    )
    parser.add_argument(
        "--checkpoint_dir",
        help="The directory to store the checkpoint files",
        default="",
    )
    parser.add_argument(
        "--checkpoint_steps",
        type=_non_neg_int,
        help="Save checkpoint every this many steps."
        "If 0, no checkpoints to save.",
        default=0,
    )
    parser.add_argument(
        "--keep_checkpoint_max",
        type=_non_neg_int,
        help="The maximum number of recent checkpoint files to keep."
        "If 0, keep all.",
        default=0,
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
        "e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1, "
        "default to worker_resource_request",
    )
    parser.add_argument(
        "--worker_pod_priority", help="Priority requested by workers"
    )
    parser.add_argument(
        "--worker_image", help="Docker image for workers", default=None
    )
    parser.add_argument("--job_name", help="Job name", required=True)
    # TODO: better logic for handling volume configs
    parser.add_argument(
        "--volume_name", help="Volume name of Network File System"
    )
    parser.add_argument(
        "--mount_path", help="Mount path in the docker container"
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        default="INFO",
        help="The logging level. Default to WARNING",
    )
    parser.add_argument(
        "--image_pull_policy",
        default="Always",
        help="Image pull policy of master and workers",
    )
    parser.add_argument(
        "--restart_policy",
        default="Never",
        help="The pod restart policy when pod crashed",
    )
    parser.add_argument(
        "--namespace",
        default="default",
        type=str,
        help="The Kubernetes namespace where ElasticDL jobs run",
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
    return parser.parse_args()
