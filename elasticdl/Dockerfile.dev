#! /bin/bash
set -e
set -x

# The data generation depends on TensorFlow >= 1.12, python 3 and pyrecordio
# library. Make sure you have them installed locally.
tmp_dir=$(mktemp -d)
../python/elasticdl/datasets/mnist/gen_data.py ${tmp_dir}/data
../python/elasticdl/datasets/cifar10/gen_data.py ${tmp_dir}/data

cp -R . ${tmp_dir}/elasticdl 

docker build -t elasticdl:dev ${tmp_dir} -f- << EOF
FROM tensorflow/tensorflow:1.13.1-py3
# For GPU version, use:
# FROM tensorflow/tensorflow:1.13.1-gpu-py3
# for pytorch, use:
# FROM pytorch/pytorch

RUN apt-get update
RUN apt-get install -y unzip curl

# Install gRPC tools in Python
RUN pip install grpcio-tools

# Install the Kubernetes Python client
RUN pip install kubernetes

# Install RecordIO and its dependency
RUN apt-get install -y libsnappy-dev && \
    pip install pyrecordio

# ElasticDL prebuilt datasets
COPY data /data

COPY elasticdl /elasticdl
WORKDIR /elasticdl

RUN make
EOF

echo "Built Docker image elasticdl:dev"
