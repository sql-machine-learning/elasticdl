ARG BASE_IMAGE=tensorflow/tensorflow:2.1.0-py3
# Use tensorflow/tensorflow:2.1.0-gpu-py3 if you want GPU-support.
ARG EXTRA_PYPI_INDEX=https://pypi.org/simple

# Stage 1: builder
FROM ${BASE_IMAGE} as builder

RUN apt update && \
    apt install -y unzip curl git software-properties-common g++ wget \
    libeigen3-dev

COPY elasticdl/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt --extra-index-url=${EXTRA_PYPI_INDEX} && rm /requirements.txt

ENV TF_PATH /tmp/tensorflow
RUN cd /tmp && git clone --depth=1 https://github.com/tensorflow/tensorflow.git

# Install Go and related tools
ARG GO_MIRROR_URL=https://dl.google.com/go
ENV GOPATH /root/go
ENV PATH /usr/local/go/bin:$GOPATH/bin:$PATH
COPY elasticdl/docker/scripts/install-go.bash /
RUN /install-go.bash ${GO_MIRROR_URL} && rm /install-go.bash

# Install protobuf and protoc
COPY elasticdl/docker/scripts/install-protobuf.bash /
RUN /install-protobuf.bash && rm /install-protobuf.bash

# Install elasticdl.org/elasticdl Go package
ENV ELASTICDLPATH /root/elasticdl
COPY . /tmp/elasticdl
RUN cd /tmp/elasticdl && \
    make -f elasticdl/Makefile && \
    cp -r /tmp/elasticdl/elasticdl/pkg $ELASTICDLPATH && \
    cd $GOPATH/pkg/mod/github.com/tensorflow && \
    go mod init github.com/tensorflow &&\
    cd $ELASTICDLPATH && \
    go mod init elasticdl.org/elasticdl && \
    go mod edit -replace github.com/tensorflow=${GOPATH}/pkg/mod/github.com/tensorflow && \
    go mod tidy && \
    go install ./... 

# Create elasticdl package
#COPY setup.py /tmp/elasticdl/setup.py
RUN cd /tmp/elasticdl && \
    python setup.py bdist_wheel && \
    cp dist/elasticdl-develop-py3-none-any.whl / && \
    cd / && rm -rf /tmp/elasticdl

# Stage 2: dev
FROM builder as dev

COPY elasticdl/requirements-dev.txt /requirements-dev.txt
RUN pip install -r /requirements-dev.txt --extra-index-url=${EXTRA_PYPI_INDEX} && rm /requirements-dev.txt

# This assumes that the data generation package is independent with the
# rest part of ElasticDL.  The generated data will be in /data.
COPY elasticdl/python/data/recordio_gen/image_label.py /var/image_label.py
RUN python /var/image_label.py --dataset mnist --fraction 0.15 \
        --records_per_shard 4096 /data

# Copy frappe dataset
COPY elasticdl/python/data/recordio_gen/frappe_recordio_gen.py /var/frappe_recordio_gen.py
RUN python /var/frappe_recordio_gen.py --data /root/.keras/datasets --output_dir /data/frappe \
    --fraction 0.05
# Copy heart dataset
COPY elasticdl/python/data/recordio_gen/heart_recordio_gen.py /var/heart_recordio_gen.py
RUN python /var/heart_recordio_gen.py --data_dir /root/.keras/datasets --output_dir /data/heart
# Copy census dataset
COPY elasticdl/python/data/recordio_gen/census_recordio_gen.py /var/census_recordio_gen.py
RUN python /var/census_recordio_gen.py --data_dir /root/.keras/datasets --output_dir /data/census

RUN rm -rf /root/.keras/datasets

# Stage 3: ci
FROM dev as ci

# Install elasticdl package
RUN pip install /elasticdl-develop-py3-none-any.whl --extra-index-url=${EXTRA_PYPI_INDEX} && \
    rm /elasticdl-develop-py3-none-any.whl


WORKDIR /
COPY model_zoo /model_zoo

# Stage 4: release
FROM ${BASE_IMAGE} as release

# Replace the Splash screen from TensorFlow image.
COPY elasticdl/docker/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

# Copy GO PS
ARG GO_PS=/root/go/bin/main
RUN mkdir -p /usr/local/go/bin
COPY --from=builder  ${GO_PS} /usr/local/go/bin
ENV PATH /usr/local/go/bin::$PATH

WORKDIR /

# Install elasticdl package
COPY --from=builder  /elasticdl-develop-py3-none-any.whl /
RUN pip install /elasticdl-develop-py3-none-any.whl --extra-index-url=${EXTRA_PYPI_INDEX} &&  \
    rm /elasticdl-develop-py3-none-any.whl

