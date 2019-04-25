import os
import inspect
import shutil

def run(model_class, train_data_dir=None, 
        num_epoch=1, minibatch_size=10, 
        record_per_task=100):
    m_path, m_file = _getModelFile()
    m_file_in_docker = "/model/" + m_file
    _buildDockerImage(m_path, m_file, m_file_in_docker)
    _submit(m_file_in_docker, model_class.__name__, train_data_dir=train_data_dir, 
            num_epoch=num_epoch, minibatch_size=minibatch_size, record_per_task=record_per_task)

def _getModelFile():
    m_file = inspect.currentframe().f_back.f_back.f_code.co_filename
    m_path = os.path.abspath(os.path.dirname(m_file))
    return m_path, m_file

def _buildDockerImage(m_path, m_file, m_file_in_docker):
    d_path = os.path.abspath(os.path.dirname(
        inspect.currentframe().f_back.f_code.co_filename))
    new_dfile = m_path + "/Dockerfile"
    shutil.copyfile(d_path + "/Dockerfile", new_dfile)

    with open(new_dfile, 'a') as df:
        df.write("COPY " + m_file + " " + m_file_in_docker)
    val = os.system('docker build -t elasticdl:dev -f Dockerfile .')

def _submit(m_file, m_class, 
            train_data_dir=None, num_epoch=1, 
            minibatch_size=10, record_per_task=100):
    cmd = "kubectl run elasticdl-master --image=elasticdl:dev --image-pull-policy=Never " \
          "--restart=Never -- python -m elasticdl.master.main --model-file={} --model-class={} " \
          "--train_data_dir={} --num_epoch={} --grads_to_wait=2 --minibatch_size={} --record_per_task={}"
    cmd = cmd.format(m_file, m_class, train_data_dir, num_epoch, minibatch_size, record_per_task)
    val = os.system(cmd)
