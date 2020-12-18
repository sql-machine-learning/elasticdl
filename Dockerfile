FROM python:3.6 as base


RUN pip install elasticdl_preprocessing --extra-index-url=https://pypi.org/simple

RUN pip install elasticdl --extra-index-url=https://pypi.org/simple
RUN /bin/bash -c 'PYTHON_PKG_PATH=$(pip3 show elasticdl | grep "Location:" | cut -d " " -f2); echo "PATH=${PYTHON_PKG_PATH}/elasticdl/go/bin:$PATH" >> /root/.bashrc_elasticdl; echo ". /root/.bashrc_elasticdl" >> /root/.bashrc'

COPY . /model_zoo
RUN pip install -r /model_zoo/requirements.txt --extra-index-url=https://pypi.org/simple

