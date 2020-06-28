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

from elasticdl_client.api import zoo_build, zoo_init, zoo_push


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
    zoo_init_parser.set_defaults(func=zoo_init)
    zoo_init_parser.add_argument(
        "--base_image", type=str, default="python:latest"
    )
    zoo_init_parser.add_argument(
        "--extra_pypi_index", type=str, required=False
    )
    zoo_init_parser.add_argument("--cluster_spec", type=str, required=False)

    # elasticdl zoo build
    zoo_build_parser = zoo_subparsers.add_parser("build")
    zoo_build_parser.set_defaults(func=zoo_build)
    zoo_build_parser.add_argument("path", type=str)
    zoo_build_parser.add_argument("--image", type=str, required=True)

    # elasticdl zoo push
    zoo_push_parser = zoo_subparsers.add_parser("push")
    zoo_push_parser.set_defaults(func=zoo_push)
    zoo_push_parser.add_argument("image", type=str)

    return parser


def main():
    parser = build_argument_parser()
    args, _ = parser.parse_known_args()
    args.func(args)


if __name__ == "__main__":
    main()
