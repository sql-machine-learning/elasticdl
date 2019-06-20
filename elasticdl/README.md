# ElasticDL: Build, Test, and Run

This document is for developers who need to know details about how ElasticDL works.


## Develop using Docker 

Before we start, please follow [this guide](./docker/README.md) to build the Docker images we need.

Developing ElasticDL often includes running unit tests, which depend on datasets in the data Docker image.  To load the datasets into a Docker volume, which will be mount to the development Docker container, please run the following command:

```bash
docker run --rm -v ElasticDLData:/data elasticdl:data
```

This command maps the `/data` directory of the container, which contains datasets downloaded when we build the Docker image, to a Docker volume named `ElasticDLData`.  A Docker volume is a directory on the host machine managed by Docker.

Then, we can run the development image, while mounting the volume `ElasticDLData` to its `/data` directory by running the following command.

```bash
docker run --rm -u $(id -u):$(id -g) -it \
    -v ElasticDLData:/data \
    -v $PWD:/work -w /work elasticdl:dev
```

Please be aware that in addition to mounting the data volume, we also mount the current directory (the root source directory) to the `/work` directory in the container.


### Build Protobuf Files

Python is an interpreted language, which means we don't need to build ElasticDL.  However, ElasticDL relies on gRPC, so we need a build process to convert the `.proto` files into Python source code.  We can do this inside the development container by running the following commands.

```bash
cd /work
make -f elasticdl/Makefile
```


### Run Unit Tests

The following commands run the unit tests disabling Kubernetes-related cases.

```bash
K8S_TESTS=False pytest elasticdl/python/tests
```

Note that, some unittests may require a running Kubernetes cluster available. To include those unittests, use:

```bash
make -f elasticdl/Makefile && pytest elasticdl/python/tests
```


### Run Pre-commit Check

We have set up pre-commit checks in the Github repo for pull requests, which can catch some Python style problems. However, to avoid waiting in the Travis CI queue, you can run the pre-commit checks in the container.

```bash
pre-commit run --files $(find elasticdl/python -name '*.py' -print0 | tr '\0' ' ')
```


## Run a Distributed Job 

### Locally and Manually

To run a distributed training job locally, we can start Docker containers manually.

The following command starts a container running the master process.  The option `-v ElasticDLData:/data` mounts the testdata volume into the container.

```
docker run --net=host -v ElasticDLData:/data --rm -it elasticdl \
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

In another terminal, run the following command to start a worker.

```
docker run --net=host -v ElasticDLData:/data --rm -it elasticdl \
    bash -c "python -m elasticdl.python.elasticdl.worker.main \
          --worker_id=1 \
          --model_file=elasticdl/python/examples/mnist_functional_api.py \
          --master_addr=localhost:50001 \
          --log_level=INFO"
```

This trains a model defined in [python/examples/mnist_functional_api.py](python/examples/mnist_functional_api.py) using the MNIST dataset for 2 epoches.


### On Kubernetes Clusters

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
