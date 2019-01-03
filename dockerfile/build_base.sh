#! /bin/bash
set -e
set -x

tmp_dir=$(mktemp -d)
../python/elasticdl/datasets/mnist/gen_data.py ${tmp_dir}/data
../python/elasticdl/datasets/cifar10/gen_data.py ${tmp_dir}/data

docker build -t elasticdl/base -t reg.docker.alibaba-inc.com/elasticdl/base ${tmp_dir} -f- << EOF
FROM tensorflow/tensorflow:1.12.0-py3

# ElasticDL Code dependencies
RUN apt-get update && apt-get -y --no-install-recommends install \
    libsnappy-dev 

RUN pip3 install \
    python-snappy \
    crc32c \
    torch torchvision

# Install pre-build RecordIO package
RUN pip3 install -I https://github.com/ElasticDL/pyrecordio/raw/develop/artifacts/recordio-0.0.1-py3-none-any.whl

# ElasticDL prebuilt datasets
COPY data /data
EOF

echo "You may push the image to registry: docker push reg.docker.alibaba-inc.com/elasticdl/base"
