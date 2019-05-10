import os
import inspect
import tempfile
import time
import getpass
from string import Template
import docker
import yaml
from kubernetes.client.apis import core_v1_api
from kubernetes import config


def run(model_class, train_data_dir=None, 
        num_epoch=1, minibatch_size=10, 
        record_per_task=100, num_worker=1, grads_to_wait=2):
    # TODO: Fix this hacky code.
    m_file = inspect.currentframe().f_back.f_code.co_filename
    m_file_in_docker = "/model/" + os.path.basename(m_file) 
    timestamp = int(round(time.time() * 1000))
    _build_docker_image(m_file, m_file_in_docker, timestamp)
    yaml_content = _generate_yaml(m_file_in_docker, model_class.__name__, train_data_dir=train_data_dir, 
            num_epoch=num_epoch, minibatch_size=minibatch_size, 
            record_per_task=record_per_task, num_worker=num_worker, 
            grads_to_wait=grads_to_wait, timestamp=timestamp)
    _submit(yaml_content)

def _build_docker_image(m_file, m_file_in_docker, timestamp, image_base="elasticdl:dev"):
    DOCKER_TEMPLATE = """
FROM {}
COPY {} {}
"""

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as df:
        df.write(DOCKER_TEMPLATE.format(image_base, m_file, m_file_in_docker))

    client = docker.APIClient(base_url='unix://var/run/docker.sock') 
    for line in client.build(dockerfile=df.name, path='.', tag='elasticdl:dev_' + str(timestamp)):
        print(str(line, encoding="utf-8"))

    # TODO: upload docker image to docker hub.

def _generate_yaml(m_file, m_class,
                   train_data_dir=None, num_epoch=1,
                   minibatch_size=10, record_per_task=100, 
                   num_worker=1, grads_to_wait=2, timestamp=1):
    YAML_TEMPLATE = """
apiVersion: v1
kind: Pod
metadata:
  name: elasticdl-master-$timestamp
  labels:
    purpose: test-command
spec:
  containers:
  - name: elasticdl-master-$timestamp
    image: elasticdl:dev_$timestamp
    command: ["python"]
    args: ["-m", "elasticdl.master.main",
          "--model-file", "$m_file",
          "--num_worker", "$num_worker",
          "--worker_image", "elasticdl:dev_$timestamp",
          "--job_name", "elasticdl-$timestamp",
          "--model-class", "$m_class",
          "--train_data_dir", "$train_data_dir",
          "--num_epoch", "$num_epoch",
          "--grads_to_wait", "$grads_to_wait",
          "--minibatch_size", "$minibatch_size",
          "--record_per_task", "$record_per_task"]
    imagePullPolicy: Never
    env:
    - name: MY_POD_IP
      valueFrom:
        fieldRef:
          fieldPath: status.podIP
  restartPolicy: Never
"""
    t = Template(YAML_TEMPLATE)
    return t.substitute(m_file=m_file, m_class=m_class, 
                train_data_dir=train_data_dir, 
                timestamp=timestamp, num_worker=num_worker, num_epoch=num_epoch,
                minibatch_size=minibatch_size, record_per_task=record_per_task,
                user=getpass.getuser(), grads_to_wait=grads_to_wait)

def _submit(yaml_content):
    config.load_kube_config()
    pod_desc = yaml.safe_load(yaml_content)
    api = core_v1_api.CoreV1Api()
    resp = api.create_namespaced_pod(body=pod_desc, namespace='default')
    print("Pod created. status='%s'" % str(resp.status))
