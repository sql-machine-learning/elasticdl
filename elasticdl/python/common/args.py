from elasticdl.python.common.log_util import default_logger as logger

MODEL_SPEC_GROUP = [
    "dataset_fn",
    "eval_metrics_fn",
    "model_def",
    "model_params",
    "optimizer",
    "loss",
    "output",
    "minibatch_size",
    "grads_to_wait",
    "num_epochs",
    "tensorboard_log_dir",
    "training_data_dir",
]

EVALUATION_GROUP = [
    "evaluation_steps",
    "evaluation_data_dir",
    "evaluation_start_delay_secs",
    "evaluation_throttle_secs",
]

PREDICTION_GROUP = ["prediction_data_dir"]

CHECKPOINT_GROUP = [
    "checkpoint_filename_for_init",
    "checkpoint_steps",
    "keep_checkpoint_max",
    "checkpoint_dir",
]

ALL_ARGS_GROUPS = [
    MODEL_SPEC_GROUP,
    EVALUATION_GROUP,
    PREDICTION_GROUP,
    CHECKPOINT_GROUP,
]


def pos_int(arg):
    res = int(arg)
    if res <= 0:
        raise ValueError("Positive integer argument required. Got %s" % res)
    return res


def non_neg_int(arg):
    res = int(arg)
    if res < 0:
        raise ValueError(
            "Non-negative integer argument required. Get %s" % res
        )
    return res


def parse_envs(arg):
    """Parse environment configs as a dict.

    Support format 'k1=v1,k2=v2,k3=v3..'. Note that comma is supported
    in value field.
    """
    envs = {}
    if not arg:
        return envs

    i = 0
    fields = arg.split("=")
    if len(fields) < 2:
        return envs
    pre_key = ""
    while i < len(fields):
        if i == 0:
            pre_key = fields[i]
        elif i == len(fields) - 1:
            envs[pre_key] = fields[i]
        else:
            r = fields[i].rfind(",")
            envs[pre_key] = fields[i][:r]
            pre_key = fields[i][r + 1 :]  # noqa: E203
        i += 1
    return envs


def add_common_params(parser):
    parser.add_argument(
        "--model_zoo",
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
        "--envs",
        type=str,
        help="Runtime environment variables. (key1=value1,key2=value2), "
        "comma is supported in value field",
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
        "--dataset_fn",
        type=str,
        default="dataset_fn",
        help="The name of the dataset function defined in the model file",
    )
    parser.add_argument(
        "--eval_metrics_fn",
        type=str,
        default="eval_metrics_fn",
        help="The name of the evaluation metrics function defined "
        "in the model file",
    )
    parser.add_argument(
        "--model_def",
        type=str,
        required=True,
        help="The import path to the model definition function/class in the "
        'model zoo, e.g. "cifar10_subclass.cifar10_subclass.CustomModel"',
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
    parser.add_argument(
        "--docker_base_url",
        help="URL to the Docker server",
        default="unix://var/run/docker.sock",
    )
    parser.add_argument(
        "--docker_tlscert", help="Path to Docker client cert", default=""
    )
    parser.add_argument(
        "--docker_tlskey", help="Path to Docker client key", default=""
    )


def add_train_params(parser):
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
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--grads_to_wait",
        type=int,
        help="Number of gradients to wait before updating model",
        default=2,
    )
    parser.add_argument(
        "--training_data_dir",
        help="Training data directory. Files should be in RecordIO format",
        default="",
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
        "--output",
        type=str,
        default="",
        help="The path to save the final trained model",
    )
    parser.add_argument(
        "--use_async",
        default=False,
        help="True for asynchronous SGD, False for synchronous SGD",
    )
    parser.add_argument(
        "--lr_staleness_modulation",
        default=False,
        help="If True, master will modulate the learning rate with staleness "
        "in asynchronous SGD",
    )
    parser.add_argument(
        "--get_model_steps",
        type=int,
        default=1,
        help="Worker will get_model from PS every these steps.",
    )


def add_evaluate_params(parser):
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


def add_predict_params(parser):
    parser.add_argument(
        "--prediction_data_dir",
        help="Prediction data directory. Files should be in RecordIO format",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_filename_for_init",
        help="The checkpoint file to initialize the training model",
        required=True,
    )


def print_args(args, groups=None):
    """
    Args:
        args: parsing results returned from `parser.parse_args`
        groups: It is a list of a list. It controls which options should be
        printed together. For example, we expect all model specifications such
        as `optimizer`, `loss` are better printed together.
        groups = [["optimizer", "loss"]]
    """

    def _get_attr(instance, attribute):
        try:
            return getattr(instance, attribute)
        except AttributeError:
            return None

    dedup = set()
    if groups:
        for group in groups:
            for element in group:
                dedup.add(element)
                logger.info("%s = %s", element, _get_attr(args, element))
    other_options = [
        (key, value)
        for (key, value) in args.__dict__.items()
        if key not in dedup
    ]
    for key, value in other_options:
        logger.info("%s = %s", key, value)
