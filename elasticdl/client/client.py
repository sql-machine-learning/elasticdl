import argparse
import os
import inspect
import tempfile
import time
import getpass
import sys
from string import Template
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
        text = line.get("stream", None)
        if text:
            sys.stdout.write(text)
            sys.stdout.flush()
    print("===== Docker Image Built =====")
    if repository != None:
        for line in client.push(image_name, stream=True, decode=True):
            print(line)

def _gen_master_def(image_name, model_file, job_name, argv):
    master_yaml = """
apiVersion: v1
kind: Pod
metadata:
  name: "elasticdl-master-{job_name}"
  labels:
    purpose: test-command
spec:
  containers:
  - name: "elasticdl-master-{job_name}"
    image: "{image_name}"
    command: ["python"]
    args: [
        "-m", "elasticdl.master.main",
        "--job_name", "{job_name}",
        "--worker_image", "{image_name}",
        "--model_file", "{m_file}"
    ]
    imagePullPolicy: IfNotPresent 
    env:
    - name: MY_POD_IP
      valueFrom:
        fieldRef:
          fieldPath: status.podIP
  restartPolicy: Never
""" .format(m_file=_m_file_in_docker(model_file), image_name=image_name, job_name=job_name)

    master_def = yaml.safe_load(master_yaml)

    # Build master arguments
    master_def['spec']['containers'][0]['args'].extend(argv)
    return master_def

def _submit(image_name, model_file, job_name, argv):
    master_def = _gen_master_def(image_name, model_file, job_name, argv)
    config.load_kube_config()
    api = core_v1_api.CoreV1Api()
    resp = api.create_namespaced_pod(body=master_def, namespace="default")
    print("Master launched. status='%s'" % str(resp.status))

def main():
    parser = argparse.ArgumentParser(description="ElasticDL Client")
    # Rewrite model_file argument and pass all other arguments to master.
    parser.add_argument("--model_file", help="Path to Model file", required=True)
    parser.add_argument("--image_base", help="Base image containing elasticdl runtime environment.", required=True)
    parser.add_argument("--repository", help="The repository to push docker image to.")
    parser.add_argument("--job_name", help="ElasticDL job name", required=True)
    args, argv = parser.parse_known_args()

    job_name = args.job_name + "-" + str(int(round(time.time() * 1000)))
    image_name = args.image_base + '_' + job_name 
    _build_docker_image(args.model_file, image_name, image_base=args.image_base,
        repository=args.repository)
    _submit(image_name, args.model_file, job_name, argv)


if __name__ == "__main__":
    main()
