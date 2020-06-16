# To build Docker images, please refer to scripts/travis/build_images.sh.
ARG BASE_IMAGE

FROM ${BASE_IMAGE} as dev
ARG EXTRA_PYPI_INDEX=https://pypi.org/simple

RUN apt-get -qq update && \
    apt-get -qq install -y unzip curl git software-properties-common g++ wget \
                       libeigen3-dev > /dev/null && \
    python -m pip install --quiet --upgrade pip

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


FROM dev as allreduce

WORKDIR /root/

# Note that pip is having issue downloading PyTorch on manylinux so we use curl
# to download it instead
RUN curl -sLo torch-1.4.0-cp36-cp36m-manylinux1_x86_64.whl \
  https://files.pythonhosted.org/packages/24/19/4804aea17cd136f1705a5e98a00618cb8f6ccc375ad8bfa437408e09d058/torch-1.4.0-cp36-cp36m-manylinux1_x86_64.whl
RUN python -m pip install --quiet torch-1.4.0-cp36-cp36m-manylinux1_x86_64.whl \
        && rm torch-1.4.0-cp36-cp36m-manylinux1_x86_64.whl

RUN git clone --depth=1 https://github.com/caicloud/ftlib.git
RUN cd /root/ftlib && python -m pip install --quiet -r requirements.txt
RUN cd /root/ftlib/ftlib/consensus/gossip && bash ./gen_shared_lib.sh
RUN cp -r /root/ftlib/ftlib /usr/local/lib/python3.6/dist-packages/ftlib
