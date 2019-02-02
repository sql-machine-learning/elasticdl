#!/bin/bash

set -ex

if [[ $# -ne 1 ]]; then
  echo "Pass in the swift-tensorflow tar.gz file"
  exit 1
fi

tmp_dir=$(mktemp -d)
cp $1 ${tmp_dir}
fname=${1##*/}

cat <<EOF > ${tmp_dir}/Dockerfile

FROM reg.docker.alibaba-inc.com/elasticdl/swift-gpu-devel

COPY ${fname} /
RUN tar xzvf /${fname} --directory=usr --strip-components=1 \
    && rm /${fname}

EOF
docker build -t s4tf-gpu ${tmp_dir}
