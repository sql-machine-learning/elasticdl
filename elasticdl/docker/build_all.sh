#!/bin/bash
# Copyright 2020 The SQLFlow Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



TF_VERSION=2.1.0

if [[ ! -d .git ]]; then
    echo "We must run this script at the root of the source tree."
    exit 1
fi

if [[ $# -eq 1 && $1 == "-gpu" ]]; then
    base_image="tensorflow/tensorflow:${TF_VERSION}-gpu-py3"
    echo "To support CUDA; all images are from " $base_image
    dev_image="elasticdl:dev-gpu"
    ci_image="elasticdl:ci-gpu"

else
    base_image="tensorflow/tensorflow:${TF_VERSION}-py3"
    dev_image="elasticdl:dev"
    ci_image="elasticdl:ci"
fi

docker build --target dev -t $dev_image -f elasticdl/docker/Dockerfile --build-arg BASE_IMAGE=$base_image .

docker build --target ci -t $ci_image -f elasticdl/docker/Dockerfile --build-arg BASE_IMAGE=$base_image .

docker build -t elasticdl:dev_allreduce -f elasticdl/docker/Dockerfile.dev_allreduce .