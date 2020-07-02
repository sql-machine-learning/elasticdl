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
import shutil

import docker
from jinja2 import Template

from elasticdl_client.common import k8s_client as k8s
from elasticdl_client.common.args import (
    build_arguments_from_parsed_result,
    parse_envs,
    wrap_python_args_with_string,
)
from elasticdl_client.common.constants import BashCommandTemplate
from elasticdl_client.common.log_utils import default_logger as logger


def init_zoo(args):
    logger.info("Create the Dockerfile for the model zoo.")

    # Copy cluster spec file to the current directory if specified
    cluster_spec_path = args.cluster_spec
    cluster_spec_name = None
    if cluster_spec_path:
        if not os.path.exists(cluster_spec_path):
            raise RuntimeError(
                "The cluster spec {} doesn't exist".format(cluster_spec_path)
            )
        shutil.copy2(cluster_spec_path, os.getcwd())
        cluster_spec_name = os.path.basename(cluster_spec_path)

    # Create the docker file
    # Build the content from the template and arguments
    tmpl_str = """\
FROM {{ BASE_IMAGE }} as base

RUN pip install elasticdl_preprocessing\
 --extra-index-url={{ EXTRA_PYPI_INDEX }}

RUN pip install elasticdl --extra-index-url={{ EXTRA_PYPI_INDEX }}
ENV PATH /usr/local/lib/python3.6/site-packages/elasticdl/go/bin:$PATH

COPY . /model_zoo
RUN pip install -r /model_zoo/requirements.txt\
 --extra-index-url={{ EXTRA_PYPI_INDEX }}

{% if CLUSTER_SPEC_NAME  %}\
COPY ./{{ CLUSTER_SPEC_NAME }} /cluster_spec/{{ CLUSTER_SPEC_NAME }}\
{% endif %}
"""
    template = Template(tmpl_str)
    docker_file_content = template.render(
        BASE_IMAGE=args.base_image,
        EXTRA_PYPI_INDEX=args.extra_pypi_index,
        CLUSTER_SPEC_NAME=cluster_spec_name,
    )

    with open("./Dockerfile", mode="w") as f:
        f.write(docker_file_content)


def build_zoo(args):
    logger.info("Build the image for the model zoo.")
    # Call docker api to build the image
    # Validate the image name schema
    client = docker.DockerClient.from_env()
    for line in client.api.build(
        dockerfile="./Dockerfile",
        path=args.path,
        rm=True,
        tag=args.image,
        decode=True,
    ):
        _print_docker_progress(line)


def push_zoo(args):
    logger.info("Push the image for the model zoo.")
    # Call docker api to push the image to remote registry
    client = docker.DockerClient.from_env()
    for line in client.api.push(args.image, stream=True, decode=True):
        _print_docker_progress(line)


def train(args):
    container_args = [
        "--worker_image",
        args.image_name,
        "--model_zoo",
        args.model_zoo,
        "--cluster_spec",
        args.cluster_spec,
    ]

    container_args.extend(
        build_arguments_from_parsed_result(
            args,
            filter_args=[
                "model_zoo",
                "cluster_spec",
                "worker_image",
                "force_use_kube_config_file",
                "func",
            ],
        )
    )

    _submit_job(args.image_name, args, container_args)


def evaluate(args):
    container_args = [
        "--worker_image",
        args.image_name,
        "--model_zoo",
        args.model_zoo,
        "--cluster_spec",
        args.cluster_spec,
    ]
    container_args.extend(
        build_arguments_from_parsed_result(
            args,
            filter_args=[
                "model_zoo",
                "cluster_spec",
                "worker_image",
                "force_use_kube_config_file",
                "func",
            ],
        )
    )

    _submit_job(args.image_name, args, container_args)


def predict(args):
    container_args = [
        "--worker_image",
        args.image_name,
        "--model_zoo",
        args.model_zoo,
        "--cluster_spec",
        args.cluster_spec,
    ]

    container_args.extend(
        build_arguments_from_parsed_result(
            args,
            filter_args=[
                "model_zoo",
                "cluster_spec",
                "worker_image",
                "force_use_kube_config_file",
                "func",
            ],
        )
    )

    _submit_job(args.image_name, args, container_args)


def _submit_job(image_name, client_args, container_args):
    client = k8s.Client(
        image_name=image_name,
        namespace=client_args.namespace,
        job_name=client_args.job_name,
        cluster_spec=client_args.cluster_spec,
        force_use_kube_config_file=client_args.force_use_kube_config_file,
    )

    container_args = wrap_python_args_with_string(container_args)

    master_client_command = (
        BashCommandTemplate.SET_PIPEFAIL
        + " python -m elasticdl.python.master.main"
    )
    container_args.insert(0, master_client_command)
    if client_args.log_file_path:
        container_args.append(
            BashCommandTemplate.REDIRECTION.format(client_args.log_file_path)
        )

    python_command = " ".join(container_args)
    container_args = ["-c", python_command]

    if client_args.yaml:
        client.dump_master_yaml(
            resource_requests=client_args.master_resource_request,
            resource_limits=client_args.master_resource_limit,
            args=container_args,
            pod_priority=client_args.master_pod_priority,
            image_pull_policy=client_args.image_pull_policy,
            restart_policy=client_args.restart_policy,
            volume=client_args.volume,
            envs=parse_envs(client_args.envs),
            yaml=client_args.yaml,
        )
        logger.info(
            "ElasticDL job %s YAML has been dumped into file %s."
            % (client_args.job_name, client_args.yaml)
        )
    else:
        client.create_master(
            resource_requests=client_args.master_resource_request,
            resource_limits=client_args.master_resource_limit,
            args=container_args,
            pod_priority=client_args.master_pod_priority,
            image_pull_policy=client_args.image_pull_policy,
            restart_policy=client_args.restart_policy,
            volume=client_args.volume,
            envs=parse_envs(client_args.envs),
        )
        logger.info(
            "ElasticDL job %s was successfully submitted. "
            "The master pod is: %s."
            % (client_args.job_name, client.get_master_pod_name())
        )


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
