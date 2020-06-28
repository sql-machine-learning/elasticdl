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

import os

import docker
from jinja2 import Template


def zoo_init(args):
    print("Create the Dockerfile for the model zoo.")

    cluster_spec_path = args.cluster_spec
    cluster_spec_name = None
    if cluster_spec_path:
        if not os.path.exists(cluster_spec_path):
            raise RuntimeError(
                "The cluster spec {} doesn't exist".format(cluster_spec_path)
            )
        cluster_spec_name = os.path.basename(cluster_spec_path)

    # Create the docker file
    # Build the content from the template and arguments
    tmpl_str = """\
FROM {{ BASE_IMAGE }} as base

RUN pip install elasticdl_preprocessing
RUN pip install elasticdl

COPY . /model_zoo
{% if EXTRA_PYPI_INDEX %}
RUN pip install -r /model_zoo/requirements.txt\
  --extra-index-url={{ EXTRA_PYPI_INDEX }}\
{% else %}\
RUN pip install -r /model_zoo/requirements.txt\
{% endif %}

{% if CLUSTER_SPEC_PATH and CLUSTER_SPEC_NAME  %}\
COPY {{ CLUSTER_SPEC_PATH }} /cluster_spec/{{ CLUSTER_SPEC_NAME }}\
{% endif %}
"""
    template = Template(tmpl_str)
    docker_file_content = template.render(
        BASE_IMAGE=args.base_image,
        EXTRA_PYPI_INDEX=args.extra_pypi_index,
        CLUSTER_SPEC_PATH=cluster_spec_path,
        CLUSTER_SPEC_NAME=cluster_spec_name,
    )

    with open("./Dockerfile", mode="w+") as f:
        f.write(docker_file_content)


def zoo_build(args):
    print("Build the image for the model zoo.")
    # Call docker api to build the image
    # Validate the image name schema
    client = _get_docker_client(
        docker_base_url=args.docker_base_url,
        docker_tlscert=args.docker_tlscert,
        docker_tlskey=args.docker_tlskey,
    )
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
    client = _get_docker_client(
        docker_base_url=args.docker_base_url,
        docker_tlscert=args.docker_tlscert,
        docker_tlskey=args.docker_tlskey,
    )

    for line in client.push(args.image, stream=True, decode=True):
        _print_docker_progress(line)


def _get_docker_client(docker_base_url, docker_tlscert, docker_tlskey):
    if docker_tlscert and docker_tlskey:
        tls_config = docker.tls.TLSConfig(
            client_cert=(docker_tlscert, docker_tlskey)
        )
        return docker.APIClient(base_url=docker_base_url, tls=tls_config)
    else:
        return docker.APIClient(base_url=docker_base_url)


def _print_docker_progress(line):
    error = line.get("error", None)
    if error:
        raise RuntimeError("Docker image build: " + error)
    stream = line.get("stream", None)
    if stream:
        print(stream, end="")
    else:
        print(line)
