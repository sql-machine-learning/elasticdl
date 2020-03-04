#!/bin/bash

TF_VERSION=2.1.0

if [[ ! -d .git ]]; then
    echo "We must run this script at the root of the source tree."
    exit -1
fi

if [[ $# -eq 1 && $1 == "-gpu" ]]; then
    base_image="tensorflow/tensorflow:${TF_VERSION}-gpu-py3"
    echo "To support CUDA; all images are from " $base_image
    image="elasticdl:gpu"
    dev_image="elasticdl:gpudev"
    ci_image="elasticdl:gpuci"

else
    base_image="tensorflow/tensorflow:${TF_VERSION}-py3"
    image="elasticdl"
    dev_image="elasticdl:dev"
    ci_image="elasticdl:ci"
fi

docker build --target dev -t $dev_image -f elasticdl/docker/Dockerfile --build-arg BASE_IMAGE=$base_image .

docker build -t $image -f elasticdl/docker/Dockerfile --build-arg BASE_IMAGE=$base_image .

docker build --target ci -t $ci_image -f elasticdl/docker/Dockerfile --build-arg BASE_IMAGE=$base_image .
