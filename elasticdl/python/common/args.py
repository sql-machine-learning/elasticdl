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

import argparse

from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl_client.common.args import (
    add_common_args_between_master_and_worker,
    add_common_params,
    add_train_params,
)
from elasticdl_client.common.constants import DistributionStrategy

MODEL_SPEC_GROUP = [
    "feed",
    "eval_metrics_fn",
    "model_def",
    "model_params",
    "optimizer",
    "loss",
    "output",
    "minibatch_size",
    "grads_to_wait",
    "num_epochs",
    "training_data",
]

EVALUATION_GROUP = [
    "evaluation_steps",
    "validation_data",
]

PREDICTION_GROUP = ["prediction_data", "prediction_outputs_processor"]

CHECKPOINT_GROUP = [
    "checkpoint_dir_for_init",
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

EXCLUDE_PRINT_ARGS = ["envs", "cluster_spec_json"]


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


def print_args(args, exclude_args=[], groups=None):
    """
    Args:
        args: parsing results returned from `parser.parse_args`
        exclude_args: the arguments which won't be printed.
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
        if key not in dedup and key not in exclude_args
    ]
    for key, value in other_options:
        logger.info("%s = %s", key, value)


def _build_master_args_parser():
    parser = argparse.ArgumentParser(description="ElasticDL Master")
    parser.add_argument(
        "--port",
        default=50001,
        type=pos_int,
        help="The listening port of master",
    )
    parser.add_argument(
        "--prediction_data",
        help="Either the data directory that contains RecordIO files "
        "or an ODPS table name used for prediction.",
        default="",
    )
    add_common_params(parser)
    add_train_params(parser)

    return parser


def _check_master_args_validity(args):
    if args.prediction_data and (args.training_data or args.validation_data):
        raise ValueError(
            "Running prediction together with training or evaluation "
            "is not supported"
        )
    if args.prediction_data and not args.checkpoint_dir_for_init:
        raise ValueError(
            "checkpoint_dir_for_init is required for running " "prediction job"
        )
    if not args.use_async and args.get_model_steps > 1:
        args.get_model_steps = 1
        logger.warning(
            "get_model_steps is set to 1 while using synchronous SGD."
        )
    if args.use_async and args.grads_to_wait > 1:
        args.grads_to_wait = 1
        logger.warning(
            "grads_to_wait is set to 1 while using asynchronous SGD."
        )
    if (
        args.num_ps_pods == 0
        and args.distribution_strategy != DistributionStrategy.ALLREDUCE
    ):
        logger.warning(
            "Set the distribution strategy to be AllReduce if ps number is 0."
        )
        args.distribution_strategy = DistributionStrategy.ALLREDUCE


def parse_master_args(master_args=None):
    parser = _build_master_args_parser()

    args, unknown_args = parser.parse_known_args(args=master_args)
    print_args(args, exclude_args=EXCLUDE_PRINT_ARGS, groups=ALL_ARGS_GROUPS)
    if unknown_args:
        logger.warning("Unknown arguments: %s", unknown_args)

    _check_master_args_validity(args)

    return args


def _build_ps_args_parser():
    parser = argparse.ArgumentParser(description="ElasticDL PS")
    parser.add_argument(
        "--ps_id", help="ID unique to the PS", type=int, required=True
    )
    parser.add_argument(
        "--port", help="Port used by the PS pod", type=int, required=True
    )
    parser.add_argument("--master_addr", help="Master ip:port")

    add_common_params(parser)
    add_train_params(parser)

    return parser


def parse_ps_args(ps_args=None):
    parser = _build_ps_args_parser()

    # TODO: add PS replica address for RPC stub creation

    args, unknown_args = parser.parse_known_args(args=ps_args)
    print_args(args, groups=ALL_ARGS_GROUPS)
    if unknown_args:
        logger.warning("Unknown arguments: %s", unknown_args)

    if args.use_async and args.grads_to_wait > 1:
        args.grads_to_wait = 1
        logger.warning(
            "grads_to_wait is set to 1 while using asynchronous SGD."
        )

    return args


def _build_worker_args_parser():
    parser = argparse.ArgumentParser(description="ElasticDL Worker")
    add_common_args_between_master_and_worker(parser)
    add_train_params(parser)
    parser.add_argument(
        "--num_workers", help="The number of workers", type=int, required=False
    )
    parser.add_argument("--job_type", help="Job type", required=True)
    parser.add_argument(
        "--prediction_outputs_processor",
        help="The name of the prediction output processor class "
        "defined in the model definition file.",
        default="PredictionOutputsProcessor",
    )
    parser.add_argument(
        "--ps_addrs",
        type=str,
        help="Addresses of parameter service pods, separated by comma",
    )

    return parser


def parse_worker_args(worker_args=None):
    parser = _build_worker_args_parser()

    if worker_args:
        worker_args = list(map(str, worker_args))
    args, unknown_args = parser.parse_known_args(args=worker_args)
    print_args(args, groups=ALL_ARGS_GROUPS)
    if unknown_args:
        logger.warning("Unknown arguments: %s", unknown_args)

    if args.distribution_strategy == DistributionStrategy.ALLREDUCE:
        logger.warning(
            "Set ps address to be empty for AllReduce distribution strategy"
        )
        args.ps_addrs = ""

    return args


def wrap_go_args_with_string(args):
    """Wrap argument values with string
    Args:
        args: list like ["--foo=3", "--bar=False"]

    Returns:
        list of string: like ["--foo='3'", "--bar='False'"]
    """
    result = []
    for value in args:
        equal_mark_index = value.index("=")
        arg_value_index = equal_mark_index + 1
        result.append(
            value[0:equal_mark_index] + "='{}'".format(value[arg_value_index:])
        )
    return result
