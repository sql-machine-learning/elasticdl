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

from elasticdl_client.api import build_zoo, init_zoo, push_zoo
from elasticdl_client.common import args


def build_argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.required = True

    # Initialize the parser for the `elasticdl zoo` commands
    zoo_parser = subparsers.add_parser("zoo")
    zoo_subparsers = zoo_parser.add_subparsers()
    zoo_subparsers.required = True

    # elasticdl zoo init
    zoo_init_parser = zoo_subparsers.add_parser("init")
    zoo_init_parser.set_defaults(func=init_zoo)
    args.add_zoo_init_arguments(parser)

    # elasticdl zoo build
    zoo_build_parser = zoo_subparsers.add_parser("build")
    zoo_build_parser.set_defaults(func=build_zoo)
    args.add_zoo_build_arguments(parser)

    # elasticdl zoo push
    zoo_push_parser = zoo_subparsers.add_parser("push")
    zoo_push_parser.set_defaults(func=push_zoo)
    args.add_zoo_push_arguments(parser)

    return parser


def main():
    parser = build_argument_parser()
    args, _ = parser.parse_known_args()
    args.func(args)


if __name__ == "__main__":
    main()
