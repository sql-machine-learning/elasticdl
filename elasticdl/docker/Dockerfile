ARG BASE_IMAGE=tensorflow/tensorflow:2.0.0b0-py3
# Use tensorflow/tensorflow:2.0.0b0-gpu-py3 if you want GPU-support.
FROM ${BASE_IMAGE}

# Replace the Splash screen from TensorFlow image.
COPY elasticdl/docker/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

RUN apt-get update && apt-get install -y unzip curl git

COPY requirements.txt /requirements.txt
ARG EXTRA_PYPI_INDEX
RUN pip install -r /requirements.txt --extra-index-url=${EXTRA_PYPI_INDEX}

WORKDIR /
ENV PYTHONPATH=/
COPY elasticdl /elasticdl
RUN make -f elasticdl/Makefile
