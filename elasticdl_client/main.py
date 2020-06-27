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


def zoo_init(args):
    print("Initialize model zoo")


def zoo_build(args):
    print("Build model zoo")


def zoo_push(args):
    print("Push model zoo")


def build_argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.required = True

    # Initialize the parser for the commands
    # elasticdl zoo init
    # elasticdl zoo build
    # elasticdl zoo push
    zoo_parser = subparsers.add_parser("zoo")
    zoo_subparsers = zoo_parser.add_subparsers()

    zoo_init_parser = zoo_subparsers.add_parser("init")
    zoo_init_parser.set_defaults(func=zoo_init)
    zoo_init_parser.add_argument(
        "--base_image", type=str, default="python:latest"
    )

    zoo_build_parser = zoo_subparsers.add_parser("build")
    zoo_build_parser.set_defaults(func=zoo_build)
    zoo_build_parser.add_argument("path", type=str)
    zoo_build_parser.add_argument("--image", type=str, required=True)

    zoo_push_parser = zoo_subparsers.add_parser("push")
    zoo_push_parser.set_defaults(func=zoo_push)
    zoo_push_parser.add_argument("image", type=str)

    return parser


def main():
    parser = build_argument_parser()
    args, _ = parser.parse_known_args()
    print(args)


if __name__ == "__main__":
    main()
