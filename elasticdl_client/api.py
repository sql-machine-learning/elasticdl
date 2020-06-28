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

import docker
from jinja2 import Template


def zoo_init(args):
    print("Create the Dockerfile for the model zoo.")
    # Create the docker file
    # Build the content from the template and arguments
    tmpl_str = """\
FROM {{ BASE_IMAGE }} as base

RUN pip install elasticdl_preprocessing
RUN pip install elasticdl

COPY . /model_zoo
{% if EXTRA_PYPI_INDEX %}
RUN pip install -r /model_zoo/requirements.txt\
  --extra-index-url={{ EXTRA_PYPI_INDEX }}
{% else %}
RUN pip install -r /model_zoo/requirements.txt
{% endif %}
"""
    template = Template(tmpl_str)
    docker_file_content = template.render(
        BASE_IMAGE=args.base_image, EXTRA_PYPI_INDEX=args.extra_pypi_index
    )

    with open("./Dockerfile", mode="w+") as f:
        f.write(docker_file_content)


def zoo_build(args):
    print("Build the image for the model zoo.")
    # Call docker api to build the image
    # Validate the image name schema
    client = docker.APIClient(base_url="unix://var/run/docker.sock")
    for line in client.build(
        dockerfile="./Dockerfile",
        path=".",
        rm=True,
        tag=args.image,
        decode=True,
    ):
        _print_docker_progress(line)


def zoo_push(args):
    print("Push the image for the model zoo.")
    # Call docker api to push the image to remote registry


def _print_docker_progress(line):
    error = line.get("error", None)
    if error:
        raise RuntimeError("Docker image build: " + error)
    stream = line.get("stream", None)
    if stream:
        print(stream)
    else:
        print(line)
