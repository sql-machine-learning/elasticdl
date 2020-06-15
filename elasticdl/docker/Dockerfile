# To build Docker images, please refer to scripts/travis/build_images.sh.
ARG BASE_IMAGE

# Stage 1: dev
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

# Stage 2: ci

FROM ${BASE_IMAGE} as ci
ARG EXTRA_PYPI_INDEX=https://pypi.org/simple

WORKDIR /

COPY elasticdl/docker/bashrc /etc/bash.bashrc
RUN chmod a+rx /etc/bash.bashrc

# Copy data
COPY --from=dev /data /data

# Copy model zoo
COPY model_zoo /model_zoo
RUN python -m pip install --quiet -r /model_zoo/requirements.txt \
        --extra-index-url=${EXTRA_PYPI_INDEX}

# Install elasticdl package
COPY --from=dev /elasticdl-develop-py3-none-any.whl /
RUN python -m pip install --quiet /elasticdl-develop-py3-none-any.whl \
        --extra-index-url=${EXTRA_PYPI_INDEX} \
    && rm /elasticdl-develop-py3-none-any.whl

# Add elasticdl_ps to system path
ENV PATH /usr/local/lib/python3.6/dist-packages/elasticdl/go/bin:$PATH

# Generate mnist checkpoint for evaluation and prediction
COPY --from=dev /elasticdl/scripts/gen_mnist_checkpoint.py /gen_mnist_checkpoint.py
RUN python /gen_mnist_checkpoint.py --checkpoint_dir=/model_zoo/test_data/mnist_ckpt

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
