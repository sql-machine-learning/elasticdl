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

from elasticdl.python.common import k8s_client as k8s
from elasticdl.python.common.args import (
    build_arguments_from_parsed_result,
    parse_envs,
    wrap_python_args_with_string,
)
from elasticdl.python.common.constants import (
    BashCommandTemplate,
    DistributionStrategy,
)
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.elasticdl.image_builder import (
    build_and_push_docker_image,
    remove_images,
)
from elasticdl.python.elasticdl.local_executor import LocalExecutor


def train(args):
    model_zoo = os.path.normpath(args.model_zoo)

    if args.distribution_strategy == DistributionStrategy.LOCAL:
        local_executor = LocalExecutor(args)
        local_executor.run()
        return

    image_pre_built = bool(args.image_name)

    image_name = (
        args.image_name
        if image_pre_built
        else build_and_push_docker_image(
            model_zoo=model_zoo,
            base_image=args.image_base,
            docker_image_repository=args.docker_image_repository,
            extra_pypi=args.extra_pypi_index,
            cluster_spec=args.cluster_spec,
            docker_base_url=args.docker_base_url,
            docker_tlscert=args.docker_tlscert,
            docker_tlskey=args.docker_tlskey,
        )
    )

    container_args = [
        "--worker_image",
        image_name,
        "--model_zoo",
        _model_zoo_in_docker(model_zoo, image_pre_built),
        "--cluster_spec",
        _cluster_spec_def_in_docker(args.cluster_spec),
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

    _submit_job(image_name, args, container_args)
    # TODO: print dashboard url after launching the master pod


def evaluate(args):
    model_zoo = os.path.normpath(args.model_zoo)

    image_pre_built = bool(args.image_name)

    image_name = (
        args.image_name
        if image_pre_built
        else build_and_push_docker_image(
            model_zoo=model_zoo,
            base_image=args.image_base,
            docker_image_repository=args.docker_image_repository,
            extra_pypi=args.extra_pypi_index,
            cluster_spec=args.cluster_spec,
            docker_base_url=args.docker_base_url,
            docker_tlscert=args.docker_tlscert,
            docker_tlskey=args.docker_tlskey,
        )
    )
    container_args = [
        "--worker_image",
        image_name,
        "--model_zoo",
        _model_zoo_in_docker(model_zoo, image_pre_built),
        "--cluster_spec",
        _cluster_spec_def_in_docker(args.cluster_spec),
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

    _submit_job(image_name, args, container_args)


def predict(args):
    model_zoo = os.path.normpath(args.model_zoo)

    image_pre_built = bool(args.image_name)

    image_name = (
        args.image_name
        if image_pre_built
        else build_and_push_docker_image(
            model_zoo=model_zoo,
            base_image=args.image_base,
            docker_image_repository=args.docker_image_repository,
            extra_pypi=args.extra_pypi_index,
            cluster_spec=args.cluster_spec,
            docker_base_url=args.docker_base_url,
            docker_tlscert=args.docker_tlscert,
            docker_tlskey=args.docker_tlskey,
        )
    )
    container_args = [
        "--worker_image",
        image_name,
        "--model_zoo",
        _model_zoo_in_docker(model_zoo, image_pre_built),
        "--cluster_spec",
        _cluster_spec_def_in_docker(args.cluster_spec),
    ]
    container_args.extend(
        build_arguments_from_parsed_result(
            args,
            filter_args=[
                "model_zoo",
                "cluster_spec",
                "worker_image",
                "force_use_kube_config_file",
            ],
        )
    )

    _submit_job(image_name, args, container_args)


def clean(args):
    if args.docker_image_repository and args.all:
        raise ValueError(
            "--docker_image_repository and --all cannot "
            "be specified at the same time"
        )
    if not (args.docker_image_repository or args.all):
        raise ValueError(
            "Either --docker_image_repository or --all "
            "needs to be configured"
        )
    remove_images(
        docker_image_repository=args.docker_image_repository,
        docker_base_url=args.docker_base_url,
        docker_tlscert=args.docker_tlscert,
        docker_tlskey=args.docker_tlskey,
    )


def _submit_job(image_name, client_args, container_args):
    client = k8s.Client(
        image_name=image_name,
        namespace=client_args.namespace,
        job_name=client_args.job_name,
        event_callback=None,
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


def _model_zoo_in_docker(model_zoo, image_pre_built):
    if image_pre_built:
        return model_zoo

    MODEL_ROOT_PATH = "/"
    return os.path.join(MODEL_ROOT_PATH, os.path.basename(model_zoo))


def _cluster_spec_def_in_docker(cluster_spec):
    CLUSTER_SPEC_ROOT_PATH = "/cluster_spec"
    return (
        os.path.join(CLUSTER_SPEC_ROOT_PATH, os.path.basename(cluster_spec))
        if cluster_spec
        else ""
    )
