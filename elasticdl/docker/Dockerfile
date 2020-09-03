# To build Docker images, please refer to scripts/travis/build_images.sh.
ARG BASE_IMAGE

FROM ${BASE_IMAGE} as dev
ARG EXTRA_PYPI_INDEX=https://pypi.org/simple

COPY elasticdl/docker/bashrc /etc/bash.bashrc
RUN chmod a+rx /etc/bash.bashrc

RUN apt-get -qq update && apt-get -qq install -y \
        unzip \
        curl \
        git \
        software-properties-common \
        g++ \
        wget \
        build-essential \
        cmake \
        vim \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        shellcheck \
        libeigen3-dev \
        clang-format > /dev/null && \
        python -m pip install --quiet --upgrade pip

COPY elasticdl_client/requirements.txt /requirements.txt
RUN python -m pip install --quiet -r /requirements.txt \
        --extra-index-url=$EXTRA_PYPI_INDEX \
        && rm /requirements.txt

COPY elasticdl_preprocessing/requirements.txt /requirements.txt
RUN python -m pip install --quiet -r /requirements.txt \
        --extra-index-url=$EXTRA_PYPI_INDEX \
        && rm /requirements.txt

COPY elasticdl_preprocessing/requirements-dev.txt /requirements-dev.txt
RUN python -m pip install --quiet -r /requirements-dev.txt \
        --extra-index-url=$EXTRA_PYPI_INDEX \
        && rm /requirements-dev.txt

COPY elasticdl/requirements.txt /requirements.txt
RUN python -m pip install --quiet -r /requirements.txt \
        --extra-index-url=$EXTRA_PYPI_INDEX \
        && rm /requirements.txt

COPY elasticdl/requirements-dev.txt /requirements-dev.txt
RUN python -m pip install --quiet -r /requirements-dev.txt \
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

# Copy gen_dataset.sh
COPY scripts/gen_dataset.sh /scripts/gen_dataset.sh
COPY elasticdl/python/data/recordio_gen/image_label.py /scripts/image_label.py
COPY elasticdl/python/data/recordio_gen/frappe_recordio_gen.py /scripts/frappe_recordio_gen.py
COPY elasticdl/python/data/recordio_gen/heart_recordio_gen.py /scripts/heart_recordio_gen.py

FROM dev as allreduce

# TODO: We should use a tag or branch name after Horovod releases
# elastic training codes.
ARG HOROVOD_COMMIT_ID="3108a24"

RUN pip install future typing

# Note that pip is having issue downloading PyTorch on manylinux so we use curl
# to download it instead
RUN curl -sLo torch-1.4.0-cp36-cp36m-manylinux1_x86_64.whl \
        https://files.pythonhosted.org/packages/24/19/4804aea17cd136f1705a5e98a00618cb8f6ccc375ad8bfa437408e09d058/torch-1.4.0-cp36-cp36m-manylinux1_x86_64.whl
RUN python -m pip install --quiet torch-1.4.0-cp36-cp36m-manylinux1_x86_64.whl \
        && rm torch-1.4.0-cp36-cp36m-manylinux1_x86_64.whl

# The latest package of Horovod does not support elastic training,
# so we need to git clone and install it using source codes.
ENV HOROVOD_PATH /tmp/horovod
RUN cd /tmp \
        && git clone -n https://github.com/horovod/horovod.git \
        && cd horovod \
        && git checkout ${HOROVOD_COMMIT_ID} \
        && git submodule update --init --recursive

RUN cd ${HOROVOD_PATH} && HOROVOD_WITHOUT_MPI=1 \
        HOROVOD_WITHOUT_MXNET=1 \
        HOROVOD_WITH_TENSORFLOW=1 \
        HOROVOD_WITH_PYTORCH=1  \
        python setup.py install

RUN rm -rf $HOROVOD_PATH
