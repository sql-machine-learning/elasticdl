# Docker Images

From this Git repo, we can build the following Docker images:

1. the data image, which contains some example datasets for unit tests, regression tests, CI, and demonstrations,
1. the development image, which contains development tools and dependencies, but not the Python source code of ElasticDL, and
1. the release image, which contains the source code of ElasticDL and its dependencies, but not development tools.

Suppose that we are going to run unit tests locally or on CI, we'd run the development image accompanied by the data image as a data volume provider.  If we are going to demonstrate on minikube, we need the release and the data image.  Or, if we are going to run a real large-scale job on Google Cloud, we need the release image with data hosted on Google Cloud storage.


## The Data Image

We can build the data image by running the following command from the root source directory.  The building process downloads datasets like MNIST and CIFAR and converts them into the RecordIO format, which is the only input file format that ElasticDL accepts when doing batch learning.

```bash
docker build \
    -t elasticdl:data \
    -f elasticdl/docker/Dockerfile.data .
```


## The Development Image

We can build the development image without GPU-support by running the following command.

```bash
docker build \
    -t elasticdl:dev \
    -f elasticdl/docker/Dockerfile.dev .
```

To enable the GPU-support, we need to specify the `BASE_IMAGE` argument as follows.

```bash
docker build \
    -t elasticdl:dev-gpu \
    -f elasticdl/docker/Dockerfile.dev \
    --build-arg BASE_IMAGE=tensorflow/tensorflow:2.0.0b0-gpu-py3 .
```

If you have difficulties downloading PyPI packages, you can provide an extra PyPI index URL in `EXTRA_PYPI_INDEX`.

```bash
docker build \
    --build-arg EXTRA_PYPI_INDEX=https://mirrors.aliyun.com/pypi/simple \
    -t elasticdl:dev \
    -f elasticdl/docker/Dockerfile.dev .
```


## The Release Image

We can build the release image without GPU-support by running the following command.

```bash
git checkout tags/v0.1 # Or any other version you want.
docker build \
    -t elasticdl \
    -f elasticdl/docker/Dockerfile .
```

Similar to how we build the development image, we can specify values of `BASE_IMAGE` to enable GPU, and/or specify `EXTRA_PYPI_INDEX` to use extra PyPI mirrors.
