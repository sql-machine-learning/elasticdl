import argparse
from itertools import chain

from elasticdl.python.common.log_utils import default_logger as logger

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
    "training_data",
]

EVALUATION_GROUP = [
    "evaluation_steps",
    "evaluation_data",
    "evaluation_start_delay_secs",
    "evaluation_throttle_secs",
]

PREDICTION_GROUP = ["prediction_data"]

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


def add_bool_param(parser, name, default, help):
    parser.add_argument(
        name,  # should be in "--foo" format
        nargs="?",
        const=not default,
        default=default,
        type=lambda x: x.lower() in ["true", "yes", "t", "y"],
        help=help,
    )


def add_common_params(parser):
    """Common arguments for training/prediction/evaluation"""
    add_common_args_between_master_and_worker(parser)
    parser.add_argument(
        "--docker_image_repository",
        default="",
        help="The repository for generated Docker images, if set, the image "
        "is also pushed to the repository",
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
        "--num_ps_pods", type=int, help="Number of PS pods", default=0
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
        "--extra_pypi_index",
        default="https://pypi.org/simple",
        help="The extra URLs of Python package repository indexes",
    )
    parser.add_argument(
        "--namespace",
        default="default",
        type=str,
        help="The name of the Kubernetes namespace where ElasticDL "
        "pods will be created",
    )
    parser.add_argument(
        "--num_minibatches_per_task",
        type=int,
        help="The number of minibatches per task",
        required=True,
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
        default=1,
    )
    parser.add_argument(
        "--training_data",
        help="Either the data directory that contains RecordIO files "
        "or an ODPS table name used for training.",
        default="",
    )
    parser.add_argument(
        "--evaluation_data",
        help="Either the data directory that contains RecordIO files "
        "or an ODPS table name used for evaluation.",
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
        "--output",
        type=str,
        default="",
        help="The path to save the final trained model",
    )

    add_bool_param(
        parser=parser,
        name="--use_async",
        default=False,
        help="True for asynchronous SGD, False for synchronous SGD",
    )
    add_bool_param(
        parser=parser,
        name="--lr_staleness_modulation",
        default=False,
        help="If True, master will modulate the learning rate with staleness "
        "in asynchronous SGD",
    )


def add_evaluate_params(parser):
    parser.add_argument(
        "--evaluation_data",
        help="Either the data directory that contains RecordIO files "
        "or an ODPS table name used for evaluation.",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_filename_for_init",
        help="The checkpoint file to initialize the training model",
        required=True,
    )


def add_predict_params(parser):
    parser.add_argument(
        "--prediction_data",
        help="Either the data directory that contains RecordIO files "
        "or an ODPS table name used for prediction.",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_filename_for_init",
        help="The checkpoint file to initialize the training model",
        required=True,
    )


def add_clean_params(parser):
    parser.add_argument(
        "--docker_image_repository",
        type=str,
        help="Clean docker images belonging to this repository.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Clean all local docker images"
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


def add_common_args_between_master_and_worker(parser):
    parser.add_argument(
        "--minibatch_size",
        help="Minibatch size for worker",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--model_zoo",
        help="The directory that contains user-defined model files "
        "or a specific model file",
        required=True,
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument(
        "--dataset_fn",
        type=str,
        default="dataset_fn",
        help="The name of the dataset function defined in the model file",
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
        help="The model parameters in a string separated by semi-colon "
        'used to instantiate the model, e.g. "param1=1; param2=2"',
    )
    parser.add_argument(
        "--get_model_steps",
        type=int,
        default=1,
        help="Worker will get_model from PS every this many steps",
    )
    parser.add_argument(
        "--data_reader_params",
        type=str,
        default="",
        help="The data reader parameters in a string separated by semi-colon "
        'used to instantiate the data reader, e.g. "param1=1; param2=2"',
    )


def parse_master_args(master_args=None):
    parser = argparse.ArgumentParser(description="ElasticDL Master")
    parser.add_argument(
        "--port",
        default=50001,
        type=pos_int,
        help="The listening port of master",
    )
    parser.add_argument(
        "--worker_image", help="Docker image for workers", default=None
    )
    parser.add_argument(
        "--worker_pod_priority", help="Priority requested by workers"
    )
    parser.add_argument(
        "--prediction_data",
        help="Either the data directory that contains RecordIO files "
        "or an ODPS table name used for prediction.",
        default="",
    )
    add_common_params(parser)
    add_train_params(parser)

    args, unknown_args = parser.parse_known_args(args=master_args)
    print_args(args, groups=ALL_ARGS_GROUPS)
    if unknown_args:
        logger.warning("Unknown arguments: %s", unknown_args)

    if all(
        v == "" or v is None
        for v in [
            args.training_data,
            args.evaluation_data,
            args.prediction_data,
        ]
    ):
        raise ValueError(
            "At least one of the data directories needs to be provided"
        )

    if args.prediction_data and (args.training_data or args.evaluation_data):
        raise ValueError(
            "Running prediction together with training or evaluation "
            "is not supported"
        )
    if args.prediction_data and not args.checkpoint_filename_for_init:
        raise ValueError(
            "checkpoint_filename_for_init is required for running "
            "prediction job"
        )
    if not args.use_async and args.get_model_steps > 1:
        args.get_model_steps = 1
        logger.warning(
            "get_model_steps is set to 1 when using synchronous SGD."
        )
    if args.use_async and args.grads_to_wait > 1:
        args.grads_to_wait = 1
        logger.warning(
            "grads_to_wait is set to 1 when using asynchronous SGD."
        )

    return args


def parse_ps_args(ps_args=None):
    parser = argparse.ArgumentParser(description="ElasticDL PS")
    add_common_params(parser)
    add_train_params(parser)
    # TODO: add PS replica address for RPC stub creation

    args, unknown_args = parser.parse_known_args(args=ps_args)
    print_args(args, groups=ALL_ARGS_GROUPS)
    if unknown_args:
        logger.warning("Unknown arguments: %s", unknown_args)
    return args


def parse_worker_args(worker_args=None):
    parser = argparse.ArgumentParser(description="ElasticDL Worker")
    add_common_args_between_master_and_worker(parser)
    parser.add_argument(
        "--worker_id", help="Id unique to the worker", type=int, required=True
    )
    parser.add_argument("--job_type", help="Job type", required=True)
    parser.add_argument("--master_addr", help="Master ip:port", required=True)
    parser.add_argument(
        "--embedding_service_endpoint",
        type=str,
        default="{}",
        help="The endpoint of embedding service, "
        "e.g. \"{'ip_0': [port_0,port_1]}\"",
    )

    args, unknown_args = parser.parse_known_args(args=worker_args)
    print_args(args, groups=ALL_ARGS_GROUPS)
    if unknown_args:
        logger.warning("Unknown arguments: %s", unknown_args)
    return args


def build_arguments_from_parsed_result(args, filter_args=None):
    """Resconstruct arguments from parsed result
    Args:
        args: result from `parser.parse_args()`
    Returns:
        list of string: ready for parser to parse,
        such as ["--foo", "3", "--bar", False]
    """
    items = vars(args).items()
    if filter_args:
        items = filter(lambda item: item[0] not in filter_args, items)

    def _str_ignore_none(s):
        if s is None:
            return s
        return str(s)

    arguments = map(_str_ignore_none, chain(*items))
    arguments = [
        "--" + k if i % 2 == 0 else k for i, k in enumerate(arguments)
    ]
    return arguments
