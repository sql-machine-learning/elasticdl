ARG BASE_IMAGE=tensorflow/tensorflow:2.0.0b1-py3
# Use tensorflow/tensorflow:2.0.0b1-gpu-py3 if you want GPU-support.
FROM ${BASE_IMAGE}

# Replace the Splash screen from TensorFlow image.
COPY elasticdl/docker/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

RUN apt-get update && apt-get install -y unzip curl git wget

# install redis 5.0.5
RUN wget http://download.redis.io/releases/redis-5.0.5.tar.gz \
    && tar xzf redis-5.0.5.tar.gz \
    && cd redis-5.0.5 \
    && make >/dev/null \
    && export PATH=$path:$(pwd)/src/

# config redis environment
ENV PATH="/redis-5.0.5/src:${PATH}"

COPY elasticdl/requirements.txt /requirements.txt
ARG EXTRA_PYPI_INDEX
RUN pip install -r /requirements.txt --extra-index-url=${EXTRA_PYPI_INDEX}

WORKDIR /
ENV PYTHONPATH=/
COPY elasticdl /elasticdl
RUN make -f elasticdl/Makefile
