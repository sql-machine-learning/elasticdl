#!/bin/bash

TF_VERSION=2.0.0b1

if [[ ! -d .git ]]; then
    echo "We must run this script at the root of the source tree."
    exit -1
fi

if [[ $# -eq 1 && $1 == "-gpu" ]]; then
    base_img="tensorflow/tensorflow:${TF_VERSION}-gpu-py3"
    echo "To support CUDA; all images are from " $base_img
else
    base_img="tensorflow/tensorflow:${TF_VERSION}-py3"
fi

docker build -t elasticdl:dev -f elasticdl/docker/Dockerfile.dev --build-arg BASE_IMAGE=$base_img .

docker build -t elasticdl -f elasticdl/docker/Dockerfile --build-arg BASE_IMAGE=$base_img .

docker build -t elasticdl:ci -f elasticdl/docker/Dockerfile.ci .
