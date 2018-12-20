#! /bin/bash
set -e
set -x

tmp_dir=$(mktemp -d)
cp Dockerfile ${tmp_dir}
../../python/elasticdl/datasets/mnist/gen_data.py ${tmp_dir}/data
docker build -t elasticdl/base -t reg.docker.alibaba-inc.com/elasticdl/base ${tmp_dir}
echo "You may push the image to registry: docker push reg.docker.alibaba-inc.com/elasticdl/base"
