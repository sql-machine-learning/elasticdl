ARG BASE_IMAGE=tensorflow/tensorflow:2.0.0-py3
# Use tensorflow/tensorflow:2.0.0-gpu-py3 if you want GPU-support.
FROM ${BASE_IMAGE}

# Replace the Splash screen from TensorFlow image.
COPY elasticdl/docker/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

RUN apt-get update && apt-get install -y unzip curl git software-properties-common

# Install redis 5.0.5
RUN add-apt-repository ppa:chris-lea/redis-server -y \
    && apt-get update \
    && apt-get install -y redis-server

COPY elasticdl/requirements.txt /requirements.txt
ARG EXTRA_PYPI_INDEX
RUN pip install -r /requirements.txt --extra-index-url=${EXTRA_PYPI_INDEX}

WORKDIR /
ENV PYTHONPATH=/
COPY elasticdl /elasticdl
RUN make -f elasticdl/Makefile
