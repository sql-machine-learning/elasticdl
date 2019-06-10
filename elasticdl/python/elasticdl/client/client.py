import argparse
import os
import tempfile
import sys
import re

import docker
import yaml

from kubernetes.client.apis import core_v1_api
from kubernetes import config


def _m_file_in_docker(model_file):
    return "/model/" + os.path.basename(model_file)


def _build_docker_image(
    m_file, image_name, image_base="elasticdl:dev", repository=None
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
    print("===== Docker Image Built =====")
    if repository is not None:
        for line in client.push(image_name, stream=True, decode=True):
            print(line)


def _gen_master_def(image_name, model_file, job_name, args, argv):
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
        "--worker_cpu_request", "{worker_cpu_request}",
        "--worker_cpu_limit", "{worker_cpu_limit}",
        "--worker_memory_request", "{worker_memory_request}",
        "--worker_memory_limit", "{worker_memory_limit}"
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
        master_cpu_limit=args.master_cpu_limit,
        master_cpu_request=args.master_cpu_request,
        master_memory_limit=args.master_memory_limit,
        master_memory_request=args.master_memory_request,
        worker_cpu_limit=args.worker_cpu_limit,
        worker_cpu_request=args.worker_cpu_request,
        worker_memory_limit=args.worker_memory_limit,
        worker_memory_request=args.worker_memory_request,
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
            [
                "--image_pull_policy",
                args.image_pull_policy,
            ]
        )

    return master_def

def _gen_evaluator_def(image_name, model_file, job_name, args, argv):
    evaluator_yaml = """
apiVersion: v1
kind: Pod
metadata:
  name: "elasticdl-evaluation-{job_name}"
  labels:
    app: elasticdl
    elasticdl_job_name: {job_name}
spec:
  containers:
  - name: "elasticdl-evaluation-{job_name}"
    image: "{image_name}"
    command: ["python"]
    args: [
        "-m", "elasticdl.python.elasticdl.evaluator.main",
        "--model_file", "{model_file}",
    ]
    imagePullPolicy: {image_pull_policy}
    resources:
      limits:
        cpu:  "{cpu_limit}"
        memory: "{memory_limit}"
      requests:
        cpu:  "{cpu_request}"
        memory: "{memory_request}"
    env:
    - name: MY_POD_IP
      valueFrom:
        fieldRef:
          fieldPath: status.podIP
  restartPolicy: Never
""".format(
        model_file=_m_file_in_docker(model_file),
        image_name=image_name,
        job_name=job_name,
        cpu_limit=args.cpu_limit,
        cpu_request=args.cpu_request,
        memory_limit=args.memory_limit,
        memory_request=args.memory_request,
        image_pull_policy=args.image_pull_policy,
    )

    evaluator_def = yaml.safe_load(evaluator_yaml)

    # Build evaluator arguments
    evaluator_def["spec"]["containers"][0]["args"].extend(argv)

    if args.pod_priority is not None:
        evaluator_def["spec"]["priorityClassName"] = args.pod_priority

    if args.volume_name is not None and args.mount_path is not None:
        persistent_volume_claim = {
            "claimName": "fileserver-claim",
            "readOnly": False,
        }
        volume = {
            "name": args.volume_name,
            "persistentVolumeClaim": persistent_volume_claim,
        }
        evaluator_def["spec"]["volumes"] = [volume]
        evaluator_def["spec"]["containers"][0]["volumeMounts"] = [
            {"mountPath": args.mount_path, "name": args.volume_name}
        ]

    return evaluator_def


def _submit_training_job(image_name, model_file, job_name, args, argv):
    master_def = _gen_master_def(image_name, model_file, job_name, args, argv)
    config.load_kube_config()
    api = core_v1_api.CoreV1Api()
    resp = api.create_namespaced_pod(body=master_def, namespace="default")
    print("Master launched. status='%s'" % str(resp.status))

def _submit_evaluation_job(image_name, model_file, job_name, args, argv):
    evaluator_def = _gen_evaluator_def(image_name, model_file, job_name, args, argv)
    config.load_kube_config()
    api = core_v1_api.CoreV1Api()
    resp = api.create_namespaced_pod(body=evaluator_def, namespace="default")
    print("Evaluator launched. status='%s'" % str(resp.status))

def _valid_cpu_spec(arg):
    regexp = re.compile("([1-9]{1})([0-9]*)m$")
    if not regexp.match(arg):
        raise ValueError("invalid cpu request spec: " + arg)
    return arg


def _valid_mem_spec(arg):
    regexp = re.compile("([1-9]{1})([0-9]*)(E|P|T|G|M|K|Ei|Pi|Ti|Gi|Mi|Ki)$")
    if not regexp.match(arg):
        raise ValueError("invalid memory request spec: " + arg)
    return arg


def main():
    parser = argparse.ArgumentParser(description="ElasticDL Client")
    # Rewrite model_file argument and pass all other arguments to master.
    parser.add_argument(
        "--model_file",
        help="Path to the model file",
        required=True,
    )
    parser.add_argument(
        "--image_base",
        help="Base image containing ElasticDL runtime environment",
        required=True,
    )
    parser.add_argument(
        "--repository",
        help="The repository to push docker image to",
    )
    parser.add_argument(
        "--job_name",
        help="ElasticDL job name",
        required=True
    )
    parser.add_argument(
        "--job_type", 
        choices=["training", "evaluation"],
        help="The type of this ElasticDL job", 
        required=True)
    parser.add_argument(
        "--master_cpu_request",
        default="100m",
        type=_valid_cpu_spec,
        help="The minimal CPU required by master in training",
    )
    parser.add_argument(
        "--master_cpu_limit",
        default="100m",
        type=_valid_cpu_spec,
        help="The maximal CPU used by master in training",
    )
    parser.add_argument(
        "--master_memory_request",
        default="1024Mi",
        type=_valid_mem_spec,
        help="The minimal memory required by master in training",
    )
    parser.add_argument(
        "--master_memory_limit",
        default="1024Mi",
        type=_valid_mem_spec,
        help="The maximal memory used by master in training",
    )
    parser.add_argument(
        "--worker_cpu_request",
        default="1000m",
        type=_valid_cpu_spec,
        help="The minimal cpu required by worker",
    )
    parser.add_argument(
        "--worker_cpu_limit",
        default="1000m",
        type=_valid_cpu_spec,
        help="The maximal cpu used by worker",
    )
    parser.add_argument(
        "--worker_memory_request",
        default="4096Mi",
        type=_valid_mem_spec,
        help="The minimal memory required by worker",
    )
    parser.add_argument(
        "--worker_memory_limit",
        default="4096Mi",
        type=_valid_mem_spec,
        help="The maximal memory used by worker",
    )
    parser.add_argument(
        "--eval_cpu_request",
        default="1000m",
        type=_valid_cpu_spec,
        help="the minimal cpu required by worker in training",
    )
    parser.add_argument(
        "--eval_cpu_limit",
        default="1000m",
        type=_valid_cpu_spec,
        help="the maximal cpu used by worker in training",
    )
    parser.add_argument(
        "--eval_memory_request",
        default="4096Mi",
        type=_valid_mem_spec,
        help="the minimal memory required by worker in training",
    )
    parser.add_argument(
        "--eval_memory_limit",
        default="4096Mi",
        type=_valid_mem_spec,
        help="the maximal memory used by worker in training",
    )
    parser.add_argument(
        "--eval_pod_priority", 
        help="the requested priority of evaluation pod"
    )
    parser.add_argument(
        "--master_pod_priority",
        help="The requested priority of master pod",
    )
    parser.add_argument(
        "--volume_name",
        help="The volume name of network file system",
    )
    parser.add_argument(
        "--mount_path",
        help="The mount path in the docker container",
    )
    parser.add_argument(
        "--image_pull_policy",
        default="Always",
        help="The image pull policy of master and worker",
    )
    args, argv = parser.parse_known_args()

    job_name = args.job_name
    image_name = args.image_base + "_" + job_name
    _build_docker_image(
        args.model_file,
        image_name,
        image_base=args.image_base,
        repository=args.repository,
    )
    if args.job_type == "training":
        _submit_training_job(image_name, args.model_file, job_name, args, argv)
    elif args.job_type == "evaluation":
        _submit_evaluation_job(image_name, args.model_file, job_name, args, argv)
    else:
       raise ValueError("invalid job type: " + args.job_type) 


if __name__ == "__main__":
    main()
