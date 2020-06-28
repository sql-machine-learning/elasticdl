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
    parser.add_argument("--base_image", type=str, default="python:latest")
    parser.add_argument("--extra_pypi_index", type=str, required=False)
    parser.add_argument(
        "--cluster_spec",
        type=str,
        help="The file that contains user-defined cluster specification,"
        "the file path can be accessed by ElasticDL client.",
        default="",
    )


def add_zoo_build_arguments(parser):
    parser.add_argument("path", type=str)
    parser.add_argument("--image", type=str, required=True)
    add_docker_arguments(parser)


def add_zoo_push_arguments(parser):
    parser.add_argument("image", type=str)
    add_docker_arguments(parser)


def add_docker_arguments(parser):
    parser.add_argument(
        "--docker_base_url", type=str, default="unix://var/run/docker.sock"
    )
    parser.add_argument("--docker_tlscert", type=str, default="")
    parser.add_argument("--docker_tlskey", type=str, default="")
