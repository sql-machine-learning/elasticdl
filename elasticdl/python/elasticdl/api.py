import os

from elasticdl.python.common import k8s_client as k8s
from elasticdl.python.common.args import (
    build_arguments_from_parsed_result,
    parse_envs,
)
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.elasticdl.image_builder import (
    build_and_push_docker_image,
    remove_images,
)


def train(args):
    model_zoo = os.path.normpath(args.model_zoo)

    image_name = build_and_push_docker_image(
        model_zoo=model_zoo,
        base_image=args.image_base,
        docker_image_repository=args.docker_image_repository,
        extra_pypi=args.extra_pypi_index,
        cluster_spec=args.cluster_spec,
        docker_base_url=args.docker_base_url,
        docker_tlscert=args.docker_tlscert,
        docker_tlskey=args.docker_tlskey,
    )

    container_args = [
        "-m",
        "elasticdl.python.master.main",
        "--worker_image",
        image_name,
        "--model_zoo",
        _model_zoo_in_docker(model_zoo),
        "--cluster_spec",
        _cluster_spec_def_in_docker(args.cluster_spec),
    ]
    container_args.extend(
        build_arguments_from_parsed_result(
            args, filter_args=["model_zoo", "cluster_spec", "worker_image"]
        )
    )

    _submit_job(image_name, args, container_args)
    # TODO: print dashboard url after launching the master pod


def evaluate(args):
    model_zoo = os.path.normpath(args.model_zoo)

    image_name = build_and_push_docker_image(
        model_zoo=model_zoo,
        base_image=args.image_base,
        docker_image_repository=args.docker_image_repository,
        extra_pypi=args.extra_pypi_index,
        cluster_spec=args.cluster_spec,
        docker_base_url=args.docker_base_url,
        docker_tlscert=args.docker_tlscert,
        docker_tlskey=args.docker_tlskey,
    )
    container_args = [
        "-m",
        "elasticdl.python.master.main",
        "--worker_image",
        image_name,
        "--model_zoo",
        _model_zoo_in_docker(model_zoo),
        "--cluster_spec",
        _cluster_spec_def_in_docker(args.cluster_spec),
    ]
    container_args.extend(
        build_arguments_from_parsed_result(
            args, filter_args=["model_zoo", "cluster_spec", "worker_image"]
        )
    )

    _submit_job(image_name, args, container_args)


def predict(args):
    model_zoo = os.path.normpath(args.model_zoo)

    image_name = build_and_push_docker_image(
        model_zoo=model_zoo,
        base_image=args.image_base,
        docker_image_repository=args.docker_image_repository,
        extra_pypi=args.extra_pypi_index,
        cluster_spec=args.cluster_spec,
        docker_base_url=args.docker_base_url,
        docker_tlscert=args.docker_tlscert,
        docker_tlskey=args.docker_tlskey,
    )
    container_args = [
        "-m",
        "elasticdl.python.master.main",
        "--worker_image",
        image_name,
        "--model_zoo",
        _model_zoo_in_docker(model_zoo),
        "--cluster_spec",
        _cluster_spec_def_in_docker(args.cluster_spec),
    ]
    container_args.extend(
        build_arguments_from_parsed_result(
            args, filter_args=["model_zoo", "cluster_spec", "worker_image"]
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
    )

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
        "ElasticDL job %s was successfully submitted. The master pod is: %s."
        % (client_args.job_name, client.get_master_pod_name())
    )


def _model_zoo_in_docker(model_zoo):
    MODEL_ROOT_PATH = "/model_zoo"
    return os.path.join(MODEL_ROOT_PATH, os.path.basename(model_zoo))


def _cluster_spec_def_in_docker(cluster_spec):
    CLUSTER_SPEC_ROOT_PATH = "/cluster_spec"
    return (
        os.path.join(CLUSTER_SPEC_ROOT_PATH, os.path.basename(cluster_spec))
        if cluster_spec
        else ""
    )
