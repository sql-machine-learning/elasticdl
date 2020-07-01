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
import sys

from elasticdl_client.api import (
    build_zoo,
    evaluate,
    init_zoo,
    predict,
    push_zoo,
    train,
)
from elasticdl_client.common import args


def build_argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.required = True

    # Initialize the parser for the `elasticdl zoo` commands
    zoo_parser = subparsers.add_parser(
        "zoo",
        help="Initialize | Build | Push a docker image for the model zoo.",
    )
    zoo_subparsers = zoo_parser.add_subparsers()
    zoo_subparsers.required = True

    # elasticdl zoo init
    zoo_init_parser = zoo_subparsers.add_parser(
        "init", help="Initialize the model zoo."
    )
    zoo_init_parser.set_defaults(func=init_zoo)
    args.add_zoo_init_arguments(zoo_init_parser)

    # elasticdl zoo build
    zoo_build_parser = zoo_subparsers.add_parser(
        "build", help="Build a docker image for the model zoo."
    )
    zoo_build_parser.set_defaults(func=build_zoo)
    args.add_zoo_build_arguments(zoo_build_parser)

    # elasticdl zoo push
    zoo_push_parser = zoo_subparsers.add_parser(
        "push",
        help="Push the docker image to a remote registry for the distributed"
        "ElasticDL job.",
    )
    zoo_push_parser.set_defaults(func=push_zoo)
    args.add_zoo_push_arguments(zoo_push_parser)

    # elasticdl train
    train_parser = subparsers.add_parser(
        "train", help="Submit a ElasticDL distributed training job"
    )
    train_parser.set_defaults(func=train)
    args.add_common_params(train_parser)
    args.add_train_params(train_parser)

    # elasticdl evaluate
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Submit a ElasticDL distributed evaluation job"
    )
    evaluate_parser.set_defaults(func=evaluate)
    args.add_common_params(evaluate_parser)
    args.add_evaluate_params(evaluate_parser)

    # elasticdl predict
    predict_parser = subparsers.add_parser(
        "predict", help="Submit a ElasticDL distributed prediction job"
    )
    predict_parser.set_defaults(func=predict)
    args.add_common_params(predict_parser)
    args.add_predict_params(predict_parser)

    return parser


def main():
    parser = build_argument_parser()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args, _ = parser.parse_known_args()
    args.func(args)


if __name__ == "__main__":
    main()
