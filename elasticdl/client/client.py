import os
import inspect
import shutil
import time
import getpass

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
    shutil.copyfile(d_path + "/Dockerfile", new_dfile)

    with open(new_dfile, 'a') as df:
        df.write("COPY " + m_file + " " + m_file_in_docker)
    val = os.system('docker build -t elasticdl:dev_' + str(timestamp) + ' -f Dockerfile .')

def _generate_yaml(m_file, m_class,
                   train_data_dir=None, num_epoch=1,
                   minibatch_size=10, record_per_task=100, timestamp=1):
    yaml_file = 'job_desc.yaml'
    yaml = open(yaml_file, 'w')
    yaml.write('apiVersion: v1\n')
    yaml.write('kind: Pod\n')
    yaml.write('metadata:\n')
    yaml.write('  name: elasticdl-master-' + str(timestamp) + '\n')
    yaml.write('  labels:\n')
    yaml.write('    purpose: test-command\n')
    yaml.write('spec:\n')
    yaml.write('  containers:\n')
    yaml.write('  - name: elasticdl-master-' + str(timestamp) + '\n')
    yaml.write('    image: elasticdl:dev_' + str(timestamp) + '\n')
    yaml.write('    command: ["python"]\n')
    yaml.write('    args: ["-m", "elasticdl.master.main", ' \
            '"--model-file", "' + m_file + '", ' \
            '"--model-class", "' + m_class + '", ' \
            '"--train_data_dir", "' + train_data_dir + '", ' \
            '"--num_epoch", "' + str(num_epoch) + '", ' \
            '"--grads_to_wait", "2", ' \
            '"--minibatch_size", "' + str(minibatch_size) + '", ' \
            '"--record_per_task", "' + str(record_per_task) + '"' \
            ']\n')
    yaml.write('    imagePullPolicy: Never\n')
    yaml.write('    volumeMounts:\n')
    yaml.write('    -  mountPath: /Users/' + getpass.getuser() + '/.minikube\n')
    yaml.write('       name: minikube-mount\n')
    yaml.write('    -  mountPath: /root/.kube\n')
    yaml.write('       name: kube-mount\n')
    yaml.write('    env:\n')
    yaml.write('    - name: MY_POD_IP\n')
    yaml.write('      valueFrom:\n')
    yaml.write('        fieldRef:\n')
    yaml.write('          fieldPath: status.podIP\n')
    yaml.write('  volumes:\n')
    yaml.write('  - name: kube-mount\n')
    yaml.write('    hostPath:\n')
    yaml.write('      path: /myhome/.kube\n')
    yaml.write('  - name: minikube-mount\n')
    yaml.write('    hostPath:\n')
    yaml.write('      path: /myhome/.minikube\n')
    yaml.write('  restartPolicy: Never\n')
    yaml.close()
    return yaml_file 

def _submit(yaml_file):
    os.system('kubectl create -f ' + yaml_file)
