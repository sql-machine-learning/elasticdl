ARG BASE_IMAGE=tensorflow/tensorflow:2.1.0-py3
# Use tensorflow/tensorflow:2.1.0-gpu-py3 if you want GPU-support.

# Multi-stage is used for dev/ci/release images
# Better to build in DOCKER_BUILDKIT by setting env: DOCKER_BUILDKIT=1
# To build dev:
# docker build --target dev -t dev_image -f this_dockerfile elasticdl_repo_dir
# To build ci:
# docker build --target ci -t ci_image -f this_dockerfile elasticdl_repo_dir

# Stage 1: dev
FROM ${BASE_IMAGE} as dev
ARG EXTRA_PYPI_INDEX=https://pypi.org/simple

RUN apt-get -qq update && \
    apt-get -qq install -y unzip curl git software-properties-common g++ wget \
                       libeigen3-dev > /dev/null && \
    python -m pip install --upgrade pip

COPY elasticdl/requirements.txt /requirements.txt
RUN python -m pip -q install -r /requirements.txt \
        --extra-index-url=$EXTRA_PYPI_INDEX \
    && rm /requirements.txt
COPY elasticdl/requirements-dev.txt /requirements-dev.txt
RUN python -m pip -q install -r /requirements-dev.txt \
        --extra-index-url=$EXTRA_PYPI_INDEX \
    && rm /requirements-dev.txt

ENV TF_PATH /tmp/tensorflow
RUN cd /tmp \
    && git clone --depth=1 --branch v2.2.0-rc0 \
           https://github.com/tensorflow/tensorflow

# Install Go and related tools
ARG GO_MIRROR_URL=https://dl.google.com/go
ENV GOPATH /root/go
ENV PATH /usr/local/go/bin:$GOPATH/bin:$PATH
COPY elasticdl/docker/scripts/install-go.bash /
RUN /install-go.bash ${GO_MIRROR_URL} && rm /install-go.bash

# Install protobuf and protoc
COPY elasticdl/docker/scripts/install-protobuf.bash /
RUN /install-protobuf.bash && rm /install-protobuf.bash

# This assumes that the data generation package is independent with the
# rest part of ElasticDL.  The generated data will be in /data.
COPY elasticdl/python/data/recordio_gen/image_label.py /var/image_label.py
RUN python /var/image_label.py --dataset mnist --fraction 0.15 \
        --records_per_shard 4096 /data

# Copy frappe dataset
COPY elasticdl/python/data/recordio_gen/frappe_recordio_gen.py \
     /var/frappe_recordio_gen.py
RUN python /var/frappe_recordio_gen.py --data /root/.keras/datasets \
    --output_dir /data/frappe \
    --fraction 0.05
# Copy heart dataset
COPY elasticdl/python/data/recordio_gen/heart_recordio_gen.py \
     /var/heart_recordio_gen.py
RUN python /var/heart_recordio_gen.py --data_dir /root/.keras/datasets \
    --output_dir /data/heart

# Install elasticdl.org/elasticdl Go package
ENV ELASTICDLPATH /root/elasticdl
COPY . /elasticdl

RUN cd /elasticdl && \
    make -f elasticdl/Makefile && \
    cp -r /elasticdl/elasticdl/go/ ${ELASTICDLPATH} && \
    cd $GOPATH/pkg/mod/github.com/tensorflow && \
    go mod init github.com/tensorflow &&\
    cd $ELASTICDLPATH && \
    go mod init elasticdl.org/elasticdl && \
    go mod edit -replace github.com/tensorflow=${GOPATH}/pkg/mod/github.com/tensorflow && \
    go get k8s.io/client-go@v0.17.0 && \
    go mod tidy && \
    go install ./...

# Create elasticdl package
RUN cd /elasticdl && \
    python setup.py -q bdist_wheel && \
    cp dist/elasticdl-develop-py3-none-any.whl /


# Stage 2: ci

FROM ${BASE_IMAGE} as ci
ARG EXTRA_PYPI_INDEX=https://pypi.org/simple

COPY elasticdl/docker/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

# Copy data
COPY --from=dev /data /data

# Copy model zoo
COPY model_zoo /model_zoo
RUN python -m pip -q install -r /model_zoo/requirements.txt \
        --extra-index-url=${EXTRA_PYPI_INDEX}

# Copy GO PS
ARG GO_PS=/root/go/bin/start_ps
RUN mkdir -p /usr/local/go/bin
COPY --from=dev ${GO_PS} /usr/local/go/bin
ENV PATH /usr/local/go/bin:$PATH

WORKDIR /

# Install elasticdl package
COPY --from=dev /elasticdl-develop-py3-none-any.whl /
RUN python -m pip -q install /elasticdl-develop-py3-none-any.whl \
        --extra-index-url=${EXTRA_PYPI_INDEX} \
    && rm /elasticdl-develop-py3-none-any.whl

# Generate mnist checkpoint for evaluation and prediction
COPY --from=dev /elasticdl/scripts/gen_mnist_checkpoint.py /gen_mnist_checkpoint.py
RUN python /gen_mnist_checkpoint.py --checkpoint_dir=/model_zoo/test_data/mnist_ckpt