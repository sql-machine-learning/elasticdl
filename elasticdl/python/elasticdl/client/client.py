import argparse
import os
import tempfile
import sys

import docker
import yaml

from kubernetes.client.apis import core_v1_api
from kubernetes import config

from elasticdl.python.elasticdl.common.k8s_utils import parse_resource


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
        "--image_base",
        help="Base image containing ElasticDL runtime environment",
        required=True,
    )
    parser.add_argument(
        "--push_image",
        action="store_true",
        help="Whether to push the newly built image to remote registry",
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


def _add_evaluate_params(parser):
    # TODO add parameters for evaluation parser..
    pass


def _train(args, argv):
    job_name = args.job_name
    image_name = args.image_base + "_" + job_name
    _build_docker_image(
        args.model_file,
        image_name,
        args.push_image,
        image_base=args.image_base,
    )
    _submit(image_name, args.model_file, job_name, args, argv)


def _evaluate(args, argv):
    # TODO implement distributed evaluation.
    raise NotImplementedError()


def _m_file_in_docker(model_file):
    return "/model/" + os.path.basename(model_file)


def _build_docker_image(
    m_file, image_name, push_image, image_base="elasticdl:dev"
):
    docker_template = """
FROM {}
COPY {} {}
"""

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as df:
        df.write(
            docker_template.format(
                image_base, m_file, _m_file_in_docker(m_file)
            )
        )

    client = docker.APIClient(base_url="unix://var/run/docker.sock")
    print("===== Building Docker Image =====")
    for line in client.build(
        dockerfile=df.name, path=".", rm=True, tag=image_name, decode=True
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


def _gen_master_def(image_name, model_file, job_name, args, argv):
    master_resource_request = parse_resource(args.master_resource_request)
    master_resource_limit = parse_resource(args.master_resource_limit)

    master_yaml = """
apiVersion: v1
kind: Pod
metadata:
  name: "elasticdl-master-{job_name}"
  labels:
    app: elasticdl
    elasticdl_job_name: {job_name}
spec:
  containers:
  - name: "elasticdl-master-{job_name}"
    image: "{image_name}"
    command: ["python"]
    args: [
        "-m", "elasticdl.python.elasticdl.master.main",
        "--job_name", "{job_name}",
        "--worker_image", "{image_name}",
        "--model_file", "{m_file}",
        "--worker_resource_request", "{worker_resource_request}",
        "--worker_resource_limit", "{worker_resource_limit}"
    ]
    imagePullPolicy: {image_pull_policy}
    resources:
      limits:
        cpu:  "{master_cpu_limit}"
        memory: "{master_memory_limit}"
      requests:
        cpu:  "{master_cpu_request}"
        memory: "{master_memory_request}"
    env:
    - name: MY_POD_IP
      valueFrom:
        fieldRef:
          fieldPath: status.podIP
  restartPolicy: Never
""".format(
        m_file=_m_file_in_docker(model_file),
        image_name=image_name,
        job_name=job_name,
        # TODO: Use resource string directly similar to
        # what's done in WorkerManager. Need to wait until
        # we switch to use k8s Python API in:
        # https://github.com/wangkuiyi/elasticdl/issues/600
        master_cpu_limit=master_resource_limit["cpu"],
        master_memory_limit=master_resource_limit["memory"],
        master_cpu_request=master_resource_request["cpu"],
        master_memory_request=master_resource_request["memory"],
        master_resource_limit=args.master_resource_limit,
        worker_resource_request=args.worker_resource_request,
        worker_resource_limit=args.worker_resource_limit,
        image_pull_policy=args.image_pull_policy,
    )

    master_def = yaml.safe_load(master_yaml)

    # Build master arguments
    master_def["spec"]["containers"][0]["args"].extend(argv)

    if args.master_pod_priority is not None:
        master_def["spec"]["priorityClassName"] = args.master_pod_priority

    if args.volume_name is not None and args.mount_path is not None:
        persistent_volume_claim = {
            "claimName": "fileserver-claim",
            "readOnly": False,
        }
        volume = {
            "name": args.volume_name,
            "persistentVolumeClaim": persistent_volume_claim,
        }
        master_def["spec"]["volumes"] = [volume]
        master_def["spec"]["containers"][0]["volumeMounts"] = [
            {"mountPath": args.mount_path, "name": args.volume_name}
        ]
        master_def["spec"]["containers"][0]["args"].extend(
            [
                "--mount_path",
                args.mount_path,
                "--volume_name",
                args.volume_name,
            ]
        )

    if args.image_pull_policy is not None:
        master_def["spec"]["containers"][0]["args"].extend(
            ["--image_pull_policy", args.image_pull_policy]
        )

    return master_def


def _submit(image_name, model_file, job_name, args, argv):
    master_def = _gen_master_def(image_name, model_file, job_name, args, argv)
    config.load_kube_config()
    api = core_v1_api.CoreV1Api()
    resp = api.create_namespaced_pod(body=master_def, namespace="default")
    print("Master launched. status='%s'" % str(resp.status))


if __name__ == "__main__":
    main()
