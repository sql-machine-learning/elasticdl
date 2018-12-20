# Contributing Guideline

We recommend running ElasticDL system locally using ElasticDL Docker image. The Docker image contains the necessary components to run ElasticDL systems, such as Tensorflow, PyTorch, etc. It also contains data processing tools for preparing input data, as well as some pre-built datasets and training scripts for testing.

## How to Build ElasticDL Docker Image

The following script will clone the git repo and build the Docker image:

```bash
# on your local MacOs or Linux machine, in your git directory
git clone https://github.com/wangkuiyi/elasticdl
# build ElasticDL Docker image
elasticdl/build_docker.sh
```

The Docker image is tagged with `elasticdl/user`.

If you made any changes to the ElasticDL code, you will need to rebuild the image. The first build may take some time, the subsequent builds should be very fast.

## How to Use ElasticDL Docker Image

The following command runs a multi-threaded training job in a container, using a pre-made user module and pre-built MNIST dataset, both of them are provided with the image.

```bash
docker run -it --rm \
    --class_name=MnistCNN \
    --runner=thread \
    --num_ps=2 \
    --num_worker=2 \
    --input=/data/mnist/train \
    /elasticdl/test/mnist.py
```

It is possible to provide your own module and data do multi-thread training locally. See following sections on how to write your own module and prepare training data. For training, mount the directory containing module and data to the container and change parameter accordingly. In the following command, it assumes your `$HOME/mnist` directory contains the module and training datasets.

```bash
docker run -it --rm -v $HOME/mnist:/work
    --class_name=MnistCNN \
    --runner=thread \
    --num_ps=2 \
    --num_worker=2 \
    --input=/work/data/mnist/train \
    /work/mnist.py
```

## How to Write Your Own Module

(TODO: details) See `test/mnist.py` in code repo for an example.

## How to Prepare your Own Training Datasets

(TODO: details) See `/python/elasticdl/datasets/` in code repo for examples.
