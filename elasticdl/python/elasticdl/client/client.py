import argparse
import os
import tempfile
import sys

import docker
from elasticdl.python.elasticdl.common import k8s_client as k8s
import shutil


def main():
    parser = argparse.ArgumentParser(
        usage="""client.py <command> [<args>]

There are all the supported commands:
train         Submit a ElasticDL distributed training job.
evaluate      Submit a ElasticDL distributed evaluation job.
"""
    )
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser("train", help="client.py train -h")
    train_parser.set_defaults(func=_train)
    _add_train_params(train_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="client.py evaluate -h"
    )
    evaluate_parser.set_defaults(func=_evaluate)
    _add_evaluate_params(evaluate_parser)

    args, argv = parser.parse_known_args()
    args.func(args, argv)


def _add_train_params(parser):
    parser.add_argument(
        "--model_file", help="Path to the model file", required=True
    )
    parser.add_argument(
        "--push_image",
        action="store_true",
        help="Whether to push the newly built image to remote registry",
    )
    parser.add_argument(
        "--image_name", help="The docker image name built by ElasticDL client"
    )
    parser.add_argument("--job_name", help="ElasticDL job name", required=True)
    parser.add_argument(
        "--master_resource_request",
        default="cpu=0.1,memory=1024Mi",
        type=str,
        help="The minimal resource required by master, "
        "e.g. cpu=0.1,memory=1024Mi,disk=1024Mi,gpu=1",
    )
    parser.add_argument(
        "--master_resource_limit",
        default="cpu=0.1,memory=1024Mi",
        type=str,
        help="The maximal resource required by master, "
        "e.g. cpu=0.1,memory=1024Mi,disk=1024Mi,gpu=1",
    )
    parser.add_argument(
        "--worker_resource_request",
        default="cpu=1,memory=4096Mi",
        type=str,
        help="The minimal resource required by worker, "
        "e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1",
    )
    parser.add_argument(
        "--worker_resource_limit",
        default="cpu=1,memory=4096Mi",
        type=str,
        help="The maximal resource required by worker, "
        "e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1",
    )
    parser.add_argument(
        "--master_pod_priority", help="The requested priority of master pod"
    )
    parser.add_argument(
        "--volume_name", help="The volume name of network file system"
    )
    parser.add_argument(
        "--mount_path", help="The mount path in the docker container"
    )
    parser.add_argument(
        "--image_pull_policy",
        default="Always",
        help="The image pull policy of master and worker",
    )
    parser.add_argument(
        "--restart_policy",
        default="Never",
        help="The pod restart policy when pod crashed",
    )
    parser.add_argument(
        "--extra_pypi_index", help="The extra python package repository"
    )


def _add_evaluate_params(parser):
    # TODO add parameters for evaluation parser..
    pass


def _train(args, argv):
    job_name = args.job_name
    _build_docker_image(
        args.model_file,
        args.image_name,
        args.push_image,
        args.extra_pypi_index,
    )
    _submit(args.image_name, args.model_file, job_name, args, argv)


def _evaluate(args, argv):
    # TODO implement distributed evaluation.
    raise NotImplementedError()


def _m_file_in_docker(model_file):
    return "/model/" + os.path.basename(model_file)


def _build_docker_image(m_file, image_name, push_image, extra_pypi_index):
    docker_template = """
FROM tensorflow/tensorflow:2.0.0b0-py3 as base

# Install gRPC tools in Python
RUN pip install grpcio-tools --extra-index-url={EXTRA_PYPI_INDEX}

# Install the Kubernetes Python client
RUN pip install kubernetes --extra-index-url={EXTRA_PYPI_INDEX}

# Install Docker python SDK
RUN pip install docker --extra-index-url={EXTRA_PYPI_INDEX}

# Install RecordIO
RUN pip install 'pyrecordio>=0.0.6' --extra-index-url={EXTRA_PYPI_INDEX}

# Install Pillow for sample data processing Spark job
RUN pip install Pillow --extra-index-url=${EXTRA_PYPI_INDEX}

ENV PYTHONPATH=/
WORKDIR /
COPY elasticdl /elasticdl
COPY elasticdl/Makefile /Makefile
RUN make
COPY {SOURCE_MODEL_FILE} {TARGET_MODEL_FILE}
"""
    with tempfile.TemporaryDirectory() as ctx_dir:
        shutil.copy(m_file, ctx_dir)
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../")
        )
        shutil.copytree(base_dir, ctx_dir + "/" + os.path.basename(base_dir))
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as df:
            df.write(
                docker_template.format(
                    SOURCE_MODEL_FILE=os.path.basename(m_file),
                    TARGET_MODEL_FILE=_m_file_in_docker(m_file),
                    EXTRA_PYPI_INDEX=extra_pypi_index,
                )
            )
        client = docker.APIClient(base_url="unix://var/run/docker.sock")
        print("===== Building Docker Image =====")
        for line in client.build(
            dockerfile=df.name,
            path=ctx_dir,
            rm=True,
            tag=image_name,
            decode=True,
        ):
            if "error" in line:
                raise RuntimeError(
                    "Docker image build failure: %s" % line["error"]
                )
            text = line.get("stream", None)
            if text:
                sys.stdout.write(text)
                sys.stdout.flush()

    if push_image:
        print("===== Pushing Docker Image =====")
        for line in client.push(image_name, stream=True, decode=True):
            print(line)


def _submit(image_name, model_file, job_name, args, argv):
    container_args = [
        "-m",
        "elasticdl.python.elasticdl.master.main",
        "--job_name",
        job_name,
        "--worker_image",
        image_name,
        "--model_file",
        _m_file_in_docker(model_file),
        "--worker_resource_request",
        args.worker_resource_request,
        "--worker_resource_limit",
        args.worker_resource_request,
    ]
    container_args.extend(["--image_pull_policy", args.image_pull_policy])
    container_args.extend(["--restart_policy", args.restart_policy])

    if all([args.volume_name, args.mount_path]):
        container_args.extend(
            [
                "--mount_path",
                args.mount_path,
                "--volume_name",
                args.volume_name,
            ]
        )
    elif any([args.volume_name, args.mount_path]):
        raise ValueError(
            "Not both of the parameters volume_name and mount_path are provided."
        )

    container_args.extend(argv)

    k8s.Client(
        image_name=image_name,
        namespace="default",
        job_name=job_name,
        event_callback=None,
    ).create_master(
        job_name,
        image_name,
        _m_file_in_docker(model_file),
        args.master_resource_request,
        args.master_resource_limit,
        args.master_pod_priority,
        args.image_pull_policy,
        args.volume_name,
        args.mount_path,
        args.restart_policy,
        container_args,
    )


if __name__ == "__main__":
    main()
