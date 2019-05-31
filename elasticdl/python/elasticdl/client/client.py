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
    m_file, image_name, image_base="elasticdl:dev",
    repository=None
):
    DOCKER_TEMPLATE = """
FROM {}
COPY {} {}
"""

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as df:
        df.write(DOCKER_TEMPLATE.format(image_base, m_file, _m_file_in_docker(m_file)))

    client = docker.APIClient(base_url="unix://var/run/docker.sock")
    print("===== Building Docker Image =====")
    for line in client.build(
        dockerfile=df.name, path=".", rm=True, tag=image_name, decode=True
    ):
        if "error" in line:
            raise RuntimeError("Docker image build failure: %s" % line["error"])
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
""" .format(m_file=_m_file_in_docker(model_file), image_name=image_name, job_name=job_name,
        master_cpu_limit=args.master_cpu_limit, master_cpu_request=args.master_cpu_request, 
        master_memory_limit=args.master_memory_limit, master_memory_request=args.master_memory_request,
        worker_cpu_limit=args.worker_cpu_limit, worker_cpu_request=args.worker_cpu_request,
        worker_memory_limit=args.worker_memory_limit, worker_memory_request=args.worker_memory_request,
        image_pull_policy=args.image_pull_policy)

    master_def = yaml.safe_load(master_yaml)

    # Build master arguments
    master_def['spec']['containers'][0]['args'].extend(argv)

    if args.master_pod_priority is not None:
        master_def['spec']['priorityClassName'] = args.master_pod_priority

    if args.volume_name is not None and args.mount_path is not None:
        persistent_volume_claim = {'claimName': 'fileserver-claim', 'readOnly': False}
        volume = {'name': args.volume_name, 'persistentVolumeClaim': persistent_volume_claim}
        master_def['spec']['volumes'] = [volume]
        master_def['spec']['containers'][0]['volumeMounts'] = [
            {'mountPath': args.mount_path, 'name': args.volume_name}]
        master_def['spec']['containers'][0]['args'].extend(['--mount_path', 
            args.mount_path, '--volume_name', args.volume_name,
            '--image_pull_policy', args.image_pull_policy])

    return master_def

def _submit(image_name, model_file, job_name, args, argv):
    master_def = _gen_master_def(image_name, model_file, job_name, args, argv)
    config.load_kube_config()
    api = core_v1_api.CoreV1Api()
    resp = api.create_namespaced_pod(body=master_def, namespace="default")
    print("Master launched. status='%s'" % str(resp.status))

def _validate_params(args):
    cpu_regexp = re.compile('([1-9]{1})([0-9]*)m$')
    memory_regexp = re.compile('([1-9]{1})([0-9]*)(E|P|T|G|M|K|Ei|Pi|Ti|Gi|Mi|Ki)$') 
    if cpu_regexp.match(args.master_cpu_request) is None:
        raise ValueError("invalid master cpu request: " + args.master_cpu_request)
    if cpu_regexp.match(args.master_cpu_limit) is None:
        raise ValueError("invalid master cpu limit: " + args.master_cpu_limit)
    if memory_regexp.match(args.master_memory_request) is None:
        raise ValueError("invalid master memory request: " + args.master_memory_request)
    if memory_regexp.match(args.master_memory_limit) is None:
        raise ValueError("invalid master memory limit: " + args.master_memory_limit)

    if cpu_regexp.match(args.worker_cpu_request) is None:
        raise ValueError("invalid worker cpu request: " + args.worker_cpu_request)
    if cpu_regexp.match(args.worker_cpu_limit) is None:
        raise ValueError("invalid worker cpu limit: " + args.worker_cpu_limit)
    if memory_regexp.match(args.worker_memory_request) is None:
        raise ValueError("invalid worker memory request: " + args.worker_memory_request)
    if memory_regexp.match(args.worker_memory_limit) is None:
        raise ValueError("invalid worker memory limit: " + args.worker_memory_limit)

def main():
    parser = argparse.ArgumentParser(description="ElasticDL Client")
    # Rewrite model_file argument and pass all other arguments to master.
    parser.add_argument("--model_file", help="Path to Model file", required=True)
    parser.add_argument("--image_base", help="Base image containing elasticdl runtime environment.", required=True)
    parser.add_argument("--repository", help="The repository to push docker image to.")
    parser.add_argument("--job_name", help="ElasticDL job name", required=True)
    parser.add_argument("--master_cpu_request", 
        default="100m",
        help="the minimal cpu required by master in training")
    parser.add_argument("--master_cpu_limit", 
        default="100m",
        help="the maximal cpu used by master in training")
    parser.add_argument("--master_memory_request", 
        default="1024Mi",
        help="the minimal memory required by master in training")
    parser.add_argument("--master_memory_limit", 
        default="1024Mi",
        help="the maximal memory used by master in training")
    parser.add_argument("--worker_cpu_request", 
        default="1000m",
        help="the minimal cpu required by worker in training")
    parser.add_argument("--worker_cpu_limit", 
        default="1000m",
        help="the maximal cpu used by worker in training")
    parser.add_argument("--worker_memory_request", 
        default="4096Mi",
        help="the minimal memory required by worker in training")
    parser.add_argument("--worker_memory_limit", 
        default="4096Mi",
        help="the maximal memory used by worker in training")
    parser.add_argument("--master_pod_priority",
        help="the requested priority of master pod")
    parser.add_argument("--volume_name",
        help="the volume name of network filesytem")
    parser.add_argument("--mount_path",
        help="the mount path in the docker container")
    parser.add_argument("--image_pull_policy",
        help="the image pull policy of master and worker")
    args, argv = parser.parse_known_args()
    _validate_params(args)

    job_name = args.job_name
    image_name = args.image_base + '_' + job_name 
    _build_docker_image(args.model_file, image_name, image_base=args.image_base,
        repository=args.repository)
    _submit(image_name, args.model_file, job_name, args, argv)


if __name__ == "__main__":
    main()
