ARG BASE_IMAGE=tensorflow/tensorflow:2.1.0-py3
# Use tensorflow/tensorflow:2.1.0-gpu-py3 if you want GPU-support.
FROM ${BASE_IMAGE}

# Replace the Splash screen from TensorFlow image.
COPY elasticdl/docker/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

RUN apt-get update && apt-get install -y unzip curl git software-properties-common libeigen3-dev

COPY elasticdl/requirements.txt /requirements.txt
ARG EXTRA_PYPI_INDEX=https://pypi.org/simple
RUN pip install -r /requirements.txt --extra-index-url=${EXTRA_PYPI_INDEX}

WORKDIR /
ENV PYTHONPATH=/
COPY elasticdl /elasticdl
RUN make -f elasticdl/Makefile python_pb
