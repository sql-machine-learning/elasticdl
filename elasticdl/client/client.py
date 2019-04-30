import os
import inspect
import shutil
import time
import getpass
from string import Template

def run(model_class, train_data_dir=None, 
        num_epoch=1, minibatch_size=10, 
        record_per_task=100):
    m_path, m_file = _getModelFile()
    m_file_in_docker = "/model/" + m_file 
    timestamp = int(round(time.time() * 1000))
    _build_docker_image(m_path, m_file, m_file_in_docker, timestamp)
    yaml_file = _generate_yaml(m_file_in_docker, model_class.__name__, train_data_dir=train_data_dir, 
            num_epoch=num_epoch, minibatch_size=minibatch_size, 
            record_per_task=record_per_task, timestamp=timestamp)
    _submit(yaml_file)

def _getModelFile():
    m_file = inspect.currentframe().f_back.f_back.f_code.co_filename
    m_path = os.path.abspath(os.path.dirname(m_file))
    return m_path, m_file

def _build_docker_image(m_path, m_file, m_file_in_docker, timestamp):
    d_path = os.path.abspath(os.path.dirname(
        inspect.currentframe().f_back.f_code.co_filename))
    new_dfile = m_path + "/Dockerfile"
    shutil.copyfile(d_path + "/../Dockerfile.dev", new_dfile)

    with open(new_dfile, 'a') as df:
        df.write("COPY " + m_file + " " + m_file_in_docker)
    val = os.system('docker build -t elasticdl:dev_' + str(timestamp) + ' -f Dockerfile .')

    # TODO: upload docker image to docker hub.

def _generate_yaml(m_file, m_class,
                   train_data_dir=None, num_epoch=1,
                   minibatch_size=10, record_per_task=100, timestamp=1):
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
           "--model-class", "$m_class",
           "--train_data_dir", "$train_data_dir",
           "--num_epoch", "$num_epoch",
           "--grads_to_wait", "2",
           "--minibatch_size", "$minibatch_size",
           "--record_per_task", "$record_per_task"]
      imagePullPolicy: Never
      volumeMounts:
      -  mountPath: /Users/$user/.minikube
         name: minikube-mount
      -  mountPath: /root/.kube
         name: kube-mount
      env:
      - name: MY_POD_IP
        valueFrom:
          fieldRef:
            fieldPath: status.podIP
    volumes:
    - name: kube-mount
      hostPath:
        path: /myhome/.kube
    - name: minikube-mount
      hostPath:
        path: /myhome/.minikube
    restartPolicy: Never
  """
  t = Template(YAML_TEMPLATE)
  yaml_file = 'job_desc.yaml'
  with open(yaml_file, "w") as yaml:
      yaml.write(t.substitute(m_file=m_file, m_class=m_class, 
          train_data_dir=train_data_dir, 
          timestamp=timestamp, num_epoch=num_epoch,
          minibatch_size=minibatch_size, record_per_task=record_per_task,
          user=getpass.getuser()))
  return yaml_file

def _submit(yaml_file):
    os.system('kubectl create -f ' + yaml_file)
