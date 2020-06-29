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
        "path", type=str, help="The path where the build context locate."
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
