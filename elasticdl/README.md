# ElasticDL: A Kubernetes-native Deep Learning Framework

## Development Docker Image

Development Docker image contains dependencies for ElasticDL development and processed demo data in RecordIO format. In repo's root directory, run the following command:

```bash
docker build \
    -t elasticdl:dev \
    -f elasticdl/docker/Dockerfile.dev .
```

To build the Docker image with GPU support, run the following command:

```bash
docker build \
    -t elasticdl:dev-gpu \
    -f elasticdl/docker/Dockerfile \
    --build-arg BASE_IMAGE=tensorflow/tensorflow:2.0.0-gpu-py3 .
```

Note that since ElasticDL depends on TensorFlow, the base image must have TensorFlow installed.

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
    -v $EDL_REPO:/edl_dir \
    -w /edl_dir \
    elasticdl:dev
```

## Continuous Integration Docker Image

Continuous integration docker image contains everything from the development docker image and the ElasticDL source code. It is  used to run continuous integration with the latest version of the source code. In repo's root directory, run the following command:

```bash
docker build \
    -t elasticdl:ci \
    -f elasticdl/docker/Dockerfile.ci .
```

## Test and Debug


### Pre-commit Check

We have set up pre-commit checks in the Github repo for pull requests, which can catch some Python style problems. However, to avoid waiting in the Travis CI queue, you can run the pre-commit checks locally:

```bash
docker run --rm -it -v $EDL_REPO:/edl_dir -w /edl_dir \
    elasticdl:dev \
    bash -c \
    "pre-commit run --files $(find elasticdl/python model_zoo -name '*.py' -print0 | tr '\0' ' ')"
```

### Unit Tests

In dev Docker container's `elasticdl` repo's root directory, do the following:

```bash
make -f elasticdl/Makefile && K8S_TESTS=False pytest elasticdl/python/tests
```

Could also start Docker container and run unit tests in a single command:

```bash
docker run --rm -u $(id -u):$(id -g) -it \
    -v $EDL_REPO:/edl_dir \
    -w /edl_dir \
    elasticdl:dev \
    bash -c "make -f elasticdl/Makefile && K8S_TESTS=False pytest elasticdl/python/tests"
```

Note that, some unit tests may require a running Kubernetes cluster available. To include those unit tests, run
the following:

```bash
make -f elasticdl/Makefile && pytest elasticdl/python/tests
```

[ODPS](https://www.alibabacloud.com/product/maxcompute)-related tests require additional environment variables. To run those tests, execute the following:

```bash
docker run --rm -it -v $PWD:/edl_dir -w /edl_dir \
    -e ODPS_PROJECT_NAME=xxx \
    -e ODPS_ACCESS_ID=xxx \
    -e ODPS_ACCESS_KEY=xxx \
    -e ODPS_ENDPOINT=xxx \
    elasticdl:dev bash -c "make -f elasticdl/Makefile && K8S_TESTS=False ODPS_TESTS=True pytest elasticdl/python/tests/odps_* elasticdl/python/tests/data_reader_test.py"
```

### Test in Docker

In a terminal, start master to distribute mnist training tasks.

```bash
docker run --net=host --rm -it -v $EDL_REPO:/edl_dir -w /edl_dir \
    elasticdl:dev \
    bash -c "python -m elasticdl.python.master.main \
          --model_zoo=model_zoo \
          --model_def=mnist_functional_api.mnist_functional_api.custom_model \
          --job_name=test \
          --training_data=/data/mnist/train \
          --validation_data=/data/mnist/test \
          --evaluation_steps=15 \
          --num_epochs=2 \
          --checkpoint_steps=2 \
          --grads_to_wait=2 \
          --minibatch_size=10 \
          --num_minibatches_per_task=10 \
          --log_level=INFO"
```

In another terminal, start a worker

```bash
docker run --net=host --rm -it -v $EDL_REPO:/edl_dir -w /edl_dir \
    elasticdl:dev \
    bash -c "python -m elasticdl.python.worker.main \
          --worker_id=1 \
          --model_zoo=model_zoo \
          --model_def=mnist_functional_api.mnist_functional_api.custom_model \
          --minibatch_size=10 \
          --job_type=training_with_evaluation \
          --master_addr=localhost:50001 \
          --log_level=INFO"
```

This will train MNIST data with a model defined in [model_zoo/mnist_functional_api/mnist_functional_api.py](../model_zoo/mnist_functional_api/mnist_functional_api.py) for 2 epoches. Note that, the master will save model checkpoints in a local directory `checkpoint_dir`.

If you get some issues related to proto definitions, please run the following command to build latest proto components.
```bash
make -f elasticdl/Makefile
```

### Test with Kubernetes

We can also test ElasticDL job in a Kubernetes cluster using the previously built [image](#development-docker-image).

First make sure the built image has been pushed to a docker registry, and then run the following command to launch the job. 
```bash
kubectl apply -f manifests/examples/elasticdl-demo-k8s.yaml
```

For running demo job in Minikube, please make sure run `eval $(minikube docker-env)` first, and then build images.
```bash
kubectl apply -f manifests/examples/elasticdl-demo-minikube.yaml
```

If you find permission error in the main pod log, e.g., `"pods is forbidden: User \"system:serviceaccount:default:default\" cannot create resource \"pods\""`, you need to grant pod-related permissions for the default user.
```bash
kubectl apply -f manifests/examples/elasticdl-rbac.yaml
```

### Test on Travis CI

All tests will be executed on [Travis CI](https://travis-ci.org/sql-machine-learning/elasticdl), which includes:
* Pre-commit checks
* Unit tests
* Integration tests

The unit tests and integration tests also contain tests running on a local Kubernetes cluster via [Minikube](https://kubernetes.io/docs/setup/learning-environment/minikube/) and tests that
require data sources from [ODPS](https://www.alibabacloud.com/product/maxcompute). Please refer to [Travis configuration file](../.travis.yml) for more details.

Note that tests related to ODPS will not be executed on pull requests created from forks since
the ODPS access information has been secured on Travis and only those who have write access can retrieve it. Developers who
have write access to this repo are encouraged to submit pull requests from branches instead of forks if any code related to ODPS
has been modified.
