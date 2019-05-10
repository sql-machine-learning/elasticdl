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
    m_file, timestamp, image_base="elasticdl:dev"
):
    DOCKER_TEMPLATE = """
FROM {}
COPY {} {}
"""

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as df:
        df.write(DOCKER_TEMPLATE.format(image_base, m_file, _m_file_in_docker(m_file)))

    client = docker.APIClient(base_url="unix://var/run/docker.sock")
    for line in client.build(
        dockerfile=df.name, path=".", rm=True, tag="elasticdl:dev_" + str(timestamp)
    ):
        print(str(line, encoding="utf-8"))

    # TODO: upload docker image to docker hub.

def _gen_master_def(model_file, argv, timestamp):
    master_yaml = """
apiVersion: v1
kind: Pod
metadata:
  name: elasticdl-master-{timestamp}
  labels:
    purpose: test-command
spec:
  containers:
  - name: elasticdl-master-{timestamp}
    image: elasticdl:dev_{timestamp}
    command: ["python"]
    args: [
        "-m", "elasticdl.master.main",
        "--worker_image", "elasticdl:dev_{timestamp}",
        "--model_file", "{m_file}"
    ]
    imagePullPolicy: Never
    env:
    - name: MY_POD_IP
      valueFrom:
        fieldRef:
          fieldPath: status.podIP
  restartPolicy: Never
""" .format(m_file=_m_file_in_docker(model_file), timestamp=timestamp)

    master_def = yaml.safe_load(master_yaml)

    # Build master arguments
    master_def['spec']['containers'][0]['args'].extend(argv)
    return master_def

def _submit(model_file, argv, timestamp):
    master_def = _gen_master_def(model_file, argv, timestamp)
    config.load_kube_config()
    api = core_v1_api.CoreV1Api()
    resp = api.create_namespaced_pod(body=master_def, namespace="default")
    print("Master launched. status='%s'" % str(resp.status))

def main():
    parser = argparse.ArgumentParser(description="ElasticDL Client")
    # Rewrite model_file argument and pass all other arguments to master.
    parser.add_argument("--model_file", help="Path to Model file", required=True)
    args, argv = parser.parse_known_args()

    timestamp = int(round(time.time() * 1000))
    _build_docker_image(args.model_file, timestamp)
    _submit(args.model_file, argv, timestamp)    


if __name__ == "__main__":
    main()
