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

from itertools import chain

from elasticdl_client.common.constants import DistributionStrategy

DEFAULT_BASE_IMAGE = "python:3.6"


def add_zoo_init_params(parser):
    parser.add_argument(
        "--base_image",
        type=str,
        default=DEFAULT_BASE_IMAGE,
        help="Base Docker image.",
    )
    parser.add_argument(
        "--extra_pypi_index",
        type=str,
        default="https://pypi.org/simple",
        help="The extra URLs of Python package repository indexes",
    )
    parser.add_argument(
        "--cluster_spec",
        type=str,
        help="The file that contains user-defined cluster specification,"
        "the file path can be accessed by ElasticDL client.",
        default="",
    )
    parser.add_argument(
        "--local_pkg_dir",
        type=str,
        help="The directory of wheel packages. The image will install wheel "
        "packages in the directory",
        default="",
    )
    parser.add_argument(
        "--model_zoo",
        help="The directory that contains user-defined model files "
        "or a specific model file.",
        required=False,
        default=".",
    )


def add_zoo_build_params(parser):
    parser.add_argument(
        "path", type=str, help="The path where the build context locates."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="The name of the docker image we are building for"
        "this model zoo.",
    )


def add_zoo_push_params(parser):
    parser.add_argument(
        "image",
        type=str,
        help="The name of the docker image for this model zoo.",
    )


def add_train_params(parser):
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
        "--validation_data",
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
        "--checkpoint_dir_for_init",
        help="The checkpoint directory to initialize the training model",
        default="",
    )
    parser.add_argument(
        "--sync_version_tolerance",
        type=int,
        help="The maximum model version difference between reported gradients "
        "and PS that synchronous SGD can accepts.",
        default=0,
    )
    parser.add_argument(
        "--log_loss_steps",
        type=int,
        help="The frequency, in number of global steps, that the global step "
        "and the loss will be logged during training.",
        default=100,
    )
    parser.add_argument(
        "--max_step",
        type=int,
        help="The maximum step to train the model",
        default=0,
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
        help="If True, PS will modulate the learning rate with staleness "
        "in asynchronous SGD",
    )
    add_bool_param(
        parser=parser,
        name="--need_elasticdl_job_service",
        default=False,
        help="If true, users use ElasticDL worker framework. "
        "Otherwise, master only launch pod manager and/or other services to "
        "provide elastic training feature to other DL framework or customized "
        "AllReduce training",
    )
    add_bool_param(
        parser=parser,
        name="--need_task_manager",
        default=True,
        help="If true, master creates a task manager for dynamic sharding. "
        "Otherwise, no task manager is created",
    )
    add_bool_param(
        parser=parser,
        name="--need_pod_manager",
        default=True,
        help="If true, master creates a pod manager to maintain the "
        "cluster for the job. Otherwise, no pod manager is created",
    )
    add_bool_param(
        parser=parser,
        name="--task_fault_tolerance",
        default=True,
        help="If true, task manager supports fault tolerance, otherwise "
        "no fault tolerance.",
    )
    add_bool_param(
        parser=parser,
        name="--relaunch_timeout_worker",
        default=False,
        help="If true, the master will detect the time of worker to "
        "execute a task and relaunch the worker if timeout",
    )
    parser.add_argument(
        "--job_command",
        help="The command executed in the pod launched by the master",
        default="",
    )
    add_bool_param(
        parser=parser,
        name="--need_tf_config",
        default=False,
        help="If true, needs to set TF_CONFIG env for ps/worker. Also "
        "need to use fixed service name for workers",
    )
    parser.add_argument(
        "--relaunch_on_worker_failure",
        type=int,
        help="The number of relaunch tries for a worker failure for "
        "PS Strategy training",
        default=1,
    )
    add_bool_param(
        parser=parser,
        name="--ps_is_critical",
        default=True,
        help="If true, ps pods are critical, and ps pod failure "
        "results in job failure.",
    )
    parser.add_argument(
        "--critical_worker_index",
        default="default",
        help="If 'default', worker0 is critical for PS strategy custom "
        "training, none for others; "
        "If 'none', all workers are non-critical; "
        "Otherwise, a list of critical worker indices such as '1:0,3:1' "
        "In each pair, the first value is the pod index and the second value "
        "is the number of allowed relaunches before becoming critical",
    )
    parser.add_argument(
        "--ps_relaunch_max_num",
        type=int,
        help="The max number of ps relaunches",
        default=1,
    )
    parser.add_argument(
        "--launch_worker_after_ps_running",
        default="default",
        help="This argument indicates if launch worker "
        "pods (execpt worker0) after all ps pods are running. "
        "If 'on', launch worker "
        "pods (execpt worker0) after all ps pods are running. "
        "If 'off', launch worker pods regardless of ps pod status "
        "If 'default', when ps.core >= 16 with PS strategy, similar "
        "to 'on', otherwise, similar to 'off'. ",
    )
    add_bool_param(
        parser=parser,
        name="--enable_automate_memory",
        default=True,
        help="If true, the master will not start non-chief workers "
        "until the chief worker reports its memory usage. The master "
        "may adjust non-chief workers' memory according to the "
        "reported memory usage.",
    )


def add_evaluate_params(parser):
    parser.add_argument(
        "--validation_data",
        help="Either the data directory that contains RecordIO files "
        "or an ODPS table name used for evaluation.",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_dir_for_init",
        help="The checkpoint directory to initialize the training model",
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
        "--prediction_outputs_processor",
        help="The name of the prediction output processor class "
        "defined in the model definition file.",
        default="PredictionOutputsProcessor",
    )
    parser.add_argument(
        "--checkpoint_dir_for_init",
        help="The checkpoint directory to initialize the training model",
        required=True,
    )


def add_common_params(parser):
    """Common arguments for training/prediction/evaluation"""
    add_common_args_between_master_and_worker(parser)
    parser.add_argument(
        "--image_name",
        type=str,
        default="",
        help="The docker image for this job.",
    )
    parser.add_argument(
        "--worker_image",
        type=str,
        default="",
        help="The docker image for workers. If not specified, "
        "it will use the value of `image_name`.",
    )
    parser.add_argument(
        "--ps_image",
        type=str,
        default="",
        help="The docker image for parameter servers. If not specified, "
        "it will use the value of `image_name`.",
    )
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
        default="",
        help="The maximal resource required by master, "
        "e.g. cpu=0.1,memory=1024Mi,disk=1024Mi,gpu=1, "
        "default to master_resource_request",
    )
    parser.add_argument(
        "--num_workers", type=int, help="Number of workers", default=2
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
        default="",
        help="The maximal resource required by worker, "
        "e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1,"
        "default to worker_resource_request",
    )
    parser.add_argument(
        "--chief_resource_request",
        default="cpu=1,memory=4096Mi",
        type=str,
        help="The minimal resource required by TensorFlow estimator, "
        " master e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1",
    )
    parser.add_argument(
        "--chief_resource_limit",
        type=str,
        default="",
        help="The maximal resource required by TensorFlow estimator, "
        "master e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1,"
        "default to chief_resource_request",
    )
    parser.add_argument(
        "--master_pod_priority",
        default="",
        help="The requested priority of master pod",
    )
    parser.add_argument(
        "--chief_pod_priority",
        default="",
        help="The requested priority of tensorflow estimator master",
    )
    parser.add_argument(
        "--worker_pod_priority",
        default="",
        help="The requested priority of worker pod, we support following"
        "configs: high/low/0.5. The 0.5 means that half"
        "worker pods have high priority, and half worker pods have"
        "low priority. The default value is low",
    )
    parser.add_argument(
        "--num_ps_pods", type=int, help="Number of PS pods", default=0
    )
    parser.add_argument(
        "--ps_resource_request",
        default="cpu=1,memory=4096Mi",
        type=str,
        help="The minimal resource required by worker, "
        "e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1",
    )
    parser.add_argument(
        "--ps_resource_limit",
        default="",
        type=str,
        help="The maximal resource required by worker, "
        "e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1,"
        "default to worker_resource_request",
    )
    parser.add_argument(
        "--ps_pod_priority",
        default="",
        help="The requested priority of PS pod",
    )
    parser.add_argument(
        "--evaluator_resource_request",
        default="cpu=1,memory=4096Mi",
        type=str,
        help="The minimal resource required by evaluator, "
        "e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1",
    )
    parser.add_argument(
        "--evaluator_resource_limit",
        default="",
        type=str,
        help="The maximal resource required by evaluator, "
        "e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1,"
        "default to evaluator_resource_request",
    )
    parser.add_argument(
        "--evaluator_pod_priority",
        default="",
        help="The requested priority of PS pod",
    )
    parser.add_argument(
        "--num_evaluators",
        type=int,
        default=0,
        help="The number of evaluator pods",
    )
    parser.add_argument(
        "--volume",
        default="",
        type=str,
        help="The Kubernetes volume information, "
        "the supported volumes are `persistentVolumeClaim` and `hostPath`,"
        'e.g. "claim_name=c1,mount_path=/path1" for `persistentVolumeClaim`,'
        '"host_path=c0,mount_path=/path0" for `hostPath`,'
        'or "host_path=c0,mount_path=/path0,type=Directory" for `hostPath`,'
        '"host_path=c0,mount_path=/path0;claim_name=c1,mount_path=/path1" for'
        "multiple volumes",
    )
    parser.add_argument(
        "--image_pull_policy",
        default="Always",
        help="The image pull policy of master and worker",
        choices=["Never", "IfNotPresent", "Always"],
    )
    parser.add_argument(
        "--restart_policy",
        default="Never",
        help="The pod restart policy when pod crashed",
        choices=["Never", "OnFailure", "Always"],
    )
    parser.add_argument(
        "--envs",
        type=str,
        default="",
        help="Runtime environment variables. (key1=value1,key2=value2), "
        "comma is supported in value field",
    )
    parser.add_argument(
        "--populate_env_names",
        type=str,
        default="",
        help="The names of environment variables which master pod populates "
        "from it to its created pods such as pservers and workers. The value "
        "can be a string or a regex expression",
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
        default=8,
    )
    parser.add_argument(
        "--cluster_spec",
        help="The file that contains user-defined cluster specification,"
        "the file path can be accessed by ElasticDL client.",
        default="",
    )
    parser.add_argument(
        "--cluster_spec_json",
        type=str,
        help="A JSON-encoded string that contains user-defined cluster"
        "specification, which is an alternate for cluster_spec to"
        "avoid using file.",
        default="",
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default="",
        help="File path for dumping ElasticDL job YAML specification. "
        "Note that, if users specify --yaml, the client wouldn't submit "
        "the job automatically, and users need to launch the job through "
        "command `kubectl create -f path_to_yaml_file`.",
    )
    add_bool_param(
        parser=parser,
        name="--force_use_kube_config_file",
        default=False,
        help="If true, force to load the cluster config from ~/.kube/config "
        "while submitting the ElasticDL job. Otherwise, if the client is in a "
        "K8S environment, load the incluster config, if not, load the kube "
        "config file.",
    )
    parser.add_argument(
        "--aux_params",
        type=str,
        default="",
        help="Auxiliary parameters for misc purposes such as debugging."
        "The auxiliary parameters in a string separated "
        'by semi-colon used to debug , e.g. "param1=1; param2=2" '
        "Supported auxiliary parameters: disable_relaunch",
    )
    parser.add_argument(
        "--log_file_path",
        type=str,
        default="",
        help="The path to save logs (e.g. stdout, stderr)",
    )


def add_common_args_between_master_and_worker(parser):
    parser.add_argument(
        "--minibatch_size",
        help="Minibatch size for worker",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--model_zoo",
        help="The directory that contains user-defined model files "
        "or a specific model file. If set `image_base`, the path should"
        "be accessed by ElasticDL client. If set `image_name`, it is"
        "the path inside this pre-built image.",
        default="",
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument(
        "--feed",
        type=str,
        default="feed",
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
        "--callbacks",
        type=str,
        default="callbacks",
        help="Optional function to add callbacks to behavior during"
        "training, evaluation and inference.",
    )
    parser.add_argument(
        "--eval_metrics_fn",
        type=str,
        default="eval_metrics_fn",
        help="The name of the evaluation metrics function defined "
        "in the model file",
    )
    parser.add_argument(
        "--custom_data_reader",
        type=str,
        default="custom_data_reader",
        help="The custom data reader defined in the model file",
    )
    parser.add_argument(
        "--model_def",
        type=str,
        default="",
        help="The import path to the model definition function/class in the "
        'model zoo, e.g. "cifar10_subclass.cifar10_subclass.CustomModel"',
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
    parser.add_argument(
        "--distribution_strategy",
        type=str,
        choices=[
            "",
            DistributionStrategy.LOCAL,
            DistributionStrategy.PARAMETER_SERVER,
            DistributionStrategy.ALLREDUCE,
            DistributionStrategy.CUSTOM,
        ],
        default=DistributionStrategy.PARAMETER_SERVER,
        help="Master will use a distribution policy on a list of devices "
        "according to the distributed strategy, "
        "e.g. 'ParameterServerStrategy', 'AllreduceStrategy', "
        "'CustomStrategy' or 'Local'",
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
        "--output",
        type=str,
        default="",
        help="The path to save the final trained model",
    )


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


def build_arguments_from_parsed_result(args, filter_args=None):
    """Reconstruct arguments from parsed result
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


def wrap_python_args_with_string(args):
    """Wrap argument values with string
    Args:
        args: list like ["--foo", "3", "--bar", False]

    Returns:
        list of string: like ["--foo", "'3'", "--bar", "'False'"]
    """
    result = []
    for value in args:
        if not value.startswith("--"):
            result.append("'{}'".format(value))
        else:
            result.append(value)
    return result
