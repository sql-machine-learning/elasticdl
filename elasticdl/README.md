# ElasticDL: A Kubernetes-native Deep Learning Framework

## The Development Docker Image

Development Docker image contains ElasticDL system code and processed demo data in RecordIO format. We first build the demo data image. This only needs to be built once.

```bash
docker build \
    -t elasticdl:data \
    -f elasticdl/docker/Dockerfile.data .
```

When building the development Docker image, the data will be copied from data image to development image. Development image needs to be rebuilt when there are code changes. In repo's root directory, run the following command:

```bash
docker build \
    -t elasticdl:dev \
    -f elasticdl/docker/Dockerfile .
```

To build the Docker image with GPU support, run the following command:

```bash
docker build \
    -t elasticdl:dev-gpu \
    -f elasticdl/docker/Dockerfile \
    --build-arg BASE_IMAGE=tensorflow/tensorflow:2.0.0b0-gpu-py3 .
```

When having difficulties downloading from the main PyPI site, you could pass an extra PyPI index url to `docker build`, such as:

```bash
docker build \
    --build-arg EXTRA_PYPI_INDEX=https://mirrors.aliyun.com/pypi/simple \
    -t elasticdl:dev \
    -f elasticdl/docker/Dockerfile .
```


To develop in the Docker container, run the following command to mount your cloned `elasticdl` git repo directory (e.g. `EDL_REPO` below) to `/elasticdl` directory in the container and start container:

```bash
EDL_REPO=<your_elasticdl_git_repo>
docker run --rm -u $(id -u):$(id -g) -it \
    -v $EDL_REPO:/v \
    -w /v \
    elasticdl:dev
```

## Test and Debug


### Pre-commit Check

We have set up pre-commit checks in the Github repo for pull requests, which can catch some Python style problems. However, to avoid waiting in the Travis CI queue, you can run the pre-commit checks locally:

```bash
docker run --rm -it -v $EDL_REPO:/v -w /v \
    elasticdl:dev \
    bash -c \
    "pre-commit run --files $(find elasticdl/python -name '*.py' -print0 | tr '\0' ' ')"
```

### Unittests

In dev Docker container's `elasticdl` repo's root directory, do the following:

```bash
make -f elasticdl/Makefile && K8S_TESTS=False pytest elasticdl/python/tests
```

Could also start Docker container and run unittests in a single command:

```bash
docker run --rm -u $(id -u):$(id -g) -it \
    -v $EDL_REPO:/v \
    -w /v \
    elasticdl:dev \
    bash -c "make -f elasticdl/Makefile && K8S_TESTS=False pytest elasticdl/python/tests"
```

Note that, some unittests may require a running Kubernetes cluster available. To include those unittests, use:

```bash
make -f elasticdl/Makefile && pytest elasticdl/python/tests
```

### Test in Docker

In a terminal, start master to distribute mnist training tasks.

```
docker run --net=host --rm -it elasticdl:dev \
    bash -c "python -m elasticdl.python.elasticdl.master.main \
          --model_file=elasticdl/python/examples/mnist_functional_api.py \
          --job_name=test \
          --training_data_dir=/data/mnist/train \
          --evaluation_data_dir=/data/mnist/test \
          --records_per_task=100 \
          --num_epochs=2 \
          --checkpoint_steps=2 \
          --grads_to_wait=2 \
          --minibatch_size=10 \
          --log_level=INFO"
```

In another terminal, start a worker

```
docker run --net=host --rm -it elasticdl:dev \
    bash -c "python -m elasticdl.python.elasticdl.worker.main \
          --worker_id=1 \
          --model_file=elasticdl/python/examples/mnist_functional_api.py \
          --master_addr=localhost:50001 \
          --log_level=INFO"
```

This will train MNIST data with a model defined in [python/examples/mnist_functional_api.py](python/examples/mnist_functional_api.py) for 2 epoches.

### Test with Kubernetes

We can also test ElasticDL job in a Kubernetes environment using the previously built [image](#the-development-docker-image).

For Minikube, run the following command to launch the job.
```bash
kubectl apply -f manifests/examples/elasticdl-demo-minikube.yaml
```
Note that in order for Minikube to use local image instead of remote registries, you need to run `eval $(minikube docker-env)` first, and then build the image following [instructions](#the-development-docker-image).

For other Kubernetes clusters, first make sure the built image has been pushed to some registries, and then run the following command to launch the job. 
```bash
kubectl apply -f manifests/examples/elasticdl-demo-k8s.yaml
```

If you find permission error in the main pod log, e.g., `"pods is forbidden: User \"system:serviceaccount:default:default\" cannot create resource \"pods\""`, you need to grant pod-related permissions for the default user.
```bash
kubectl apply -f manifests/examples/elasticdl-rbac.yaml
```
