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


def add_zoo_init_arguments(parser):
    parser.add_argument(
        "--base_image",
        type=str,
        default="python:latest",
        help="Base Docker image.",
    )
    parser.add_argument(
        "--extra_pypi_index",
        type=str,
        help="The extra URLs of Python package repository indexes",
        required=False,
    )
    parser.add_argument(
        "--cluster_spec",
        type=str,
        help="The file that contains user-defined cluster specification,"
        "the file path can be accessed by ElasticDL client.",
        default="",
    )


def add_zoo_build_arguments(parser):
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
    add_docker_arguments(parser)


def add_zoo_push_arguments(parser):
    parser.add_argument(
        "image",
        type=str,
        help="The name of the docker image for this model zoo.",
    )
    add_docker_arguments(parser)


def add_docker_arguments(parser):
    parser.add_argument(
        "--docker_base_url",
        type=str,
        help="URL to the Docker server",
        default="unix://var/run/docker.sock",
    )
    parser.add_argument(
        "--docker_tlscert",
        type=str,
        help="Path to Docker client cert",
        default="",
    )
    parser.add_argument(
        "--docker_tlskey",
        type=str,
        help="Path to Docker client key",
        default="",
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
    pass


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
        if "--" not in value:
            result.append("'{}'".format(value))
        else:
            result.append(value)
    return result
