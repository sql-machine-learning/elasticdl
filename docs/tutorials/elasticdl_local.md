# ElasticDL on Personal Computer

This document shows how to run ElasticDL jobs on your personal computer using
Minikube.

## Prerequisites

1. Install Minikube, preferably >= v1.11.0, following the installation
   [guide](https://kubernetes.io/docs/tasks/tools/install-minikube).  Minikube
   runs a single-node Kubernetes cluster in a virtual machine on your personal
   computer.

1. Install Docker CE, preferably >= 18.x, following the
   [guide](https://docs.docker.com/docker-for-mac/install/) for building Docker
   images containing user-defined models and the ElasticDL framework.

1. Install Python, preferably >= 3.6, because the ElasticDL command-line tool is
   in Python.

## Models

Among all machine learning toolkits that ElasticDL can work with, TensorFlow is
the most tested and used.  In this tutorial, we use a model from the [model
zoo](https://github.com/sql-machine-learning/elasticdl/tree/develop/model_zoo)
directory.  This model is defined using TensorFlow Keras API.  To write your
models, please refer to this [tutorial](./model_contribution.md).

## Datasets

We use the MNIST dataset in this tutorial.  The dynamic data partitioning
mechanism of ElasticDL requires that the training data files are in the
[RecordIO format](https://pypi.org/project/pyrecordio). To download the MNIST
dataset and convert it into RecordIO files, please run the following command.

```bash
docker run --rm -it \
  -v $HOME/.keras:/root/.keras \
  -v $PWD:/work \
  -w /work \
  elasticdl/elasticdl:dev bash -c "/scripts/gen_dataset.sh data"
```

After the running of this command, we will see the generated dataset files in
the directory `./data`.

## The Kubernetes Cluster

The following command starts a Kubernetes cluster locally using Minikube.  It
uses [hyperkit](https://github.com/moby/hyperkit), a hypervisor coming with
macOS, to create the virtual machine cluster.  If you want, please feel free to
use other hypervisors including VirtualBox.

```bash
minikube start --vm-driver=hyperkit \
  --cpus 2 --memory 6144 --disk-size=50gb \
  --mount=true --mount-string="./data:/data"
eval $(minikube docker-env)
```

The command-line option `--mount-string` exposes the directory `./data` on the
host to Minikube as `/data`, which, we can later bind mount into containers
running on the Kubernetes cluster.

The command `minikube docker-env` returns a set of Bash environment variable
exports to configure your local environment to re-use the Docker daemon inside
the Minikube instance.

The following command is necessary to enable
[RBAC](https://kubernetes.io/docs/reference/access-authn-authz/rbac/) of
Kubernetes.

```bash
kubectl apply -f \
  https://raw.githubusercontent.com/sql-machine-learning/elasticdl/develop/elasticdl/manifests/elasticdl-rbac.yaml
```

If you happen to live in a region where `raw.githubusercontent.com` is banned,
you might want to Git clone the above repository to get the YAML file.

## Install ElasticDL Client Tool

The following command installs the command line tool `elasticdl`, which talks to
the Kubernetes cluster and operates ElasticDL jobs.

```bash
pip install elasticdl_client
```

## Build the Docker Image with Model Definition

Kubernetes runs Docker containers, so we need to put the training system,
consisting of user-defined models, ElasticDL the trainer, and all dependencies,
into a Docker image.

In this tutorial, we use a predefined model in the ElasticDL repository.  To
retrieve the source code, please run the following command.

```bash
git clone https://github.com/sql-machine-learning/elasticdl
```

Model definitions are in directory `elasticdl/model_zoo`.

### Build the Docker Image for Parameter Server

The following commands build the Docker image `elasticdl:mnist_ps`

```bash
cd elasticdl
elasticdl zoo init --model_zoo=model_zoo
elasticdl zoo build --image=elasticdl:mnist_ps .
```

### Build the Docker Image for AllReduce

We have not released ElasticDL packages with AllReduce yet. Thus,
we need to manually build packages with AllReduce support.

We must build an image `elasticdl:dev_allreduce` first using the

```bash
scripts/travis/build_images.sh
```

Then we use this image to build packages with AllReduce support.

```bash
scripts/docker_build_wheel.sh
```

After this, we can build the AllReduce training image `elasticdl:mnist_allreduce`
with model definitions in model_zoo.

```bash
elasticdl zoo init \
  --base_image=elasticdl:dev_allreduce \
  --model_zoo=model_zoo \
  --local_pkg_dir=./build
elasticdl zoo build --image=elasticdl:mnist_allreduce .
```

## Submit the Training Job Using Parameter Server

The following command submits a training job:

```bash
elasticdl train \
  --image_name=elasticdl:mnist_ps \
  --model_zoo=model_zoo \
  --model_def=mnist.mnist_functional_api.custom_model \
  --training_data=/data/mnist/train \
  --validation_data=/data/mnist/test \
  --num_epochs=2 \
  --master_resource_request="cpu=0.2,memory=1024Mi" \
  --master_resource_limit="cpu=1,memory=2048Mi" \
  --worker_resource_request="cpu=0.4,memory=1024Mi" \
  --worker_resource_limit="cpu=1,memory=2048Mi" \
  --ps_resource_request="cpu=0.2,memory=1024Mi" \
  --ps_resource_limit="cpu=1,memory=2048Mi" \
  --minibatch_size=64 \
  --num_minibatches_per_task=2 \
  --num_ps_pods=1 \
  --num_workers=1 \
  --evaluation_steps=50 \
  --job_name=test-mnist \
  --image_pull_policy=Never \
  --volume="host_path=/data,mount_path=/data" \
  --distribution_strategy=ParameterServerStrategy
```

We had exposed the directory `./data` to Minikube in above sections.  Here, the
option `--volume="host_path=/data,mount_path=/data"` bind mount it into the
containers/pods.

The above command starts a Kubernetes job with only one container, or pod, which
are exchangeable in this document), the master container.

The option `--num_workers=1` tells the master container to start a worker pod.

The option `--distribution_strategy=ParameterServerStrategy` chooses the
parameter server for the distributed stochastic gradient descent (SGD)
algorithm. The option `--num_ps_pods=1` tells the master to start one parameter
server pod. For more details about parameter server strategy, please refer to
the [design doc](/docs/designs/parameter_server.md).

### Check Job Status

After the job submission, we can run the command `kubectl get pods` to list
related containers.

```bash
NAME                            READY   STATUS    RESTARTS   AGE
elasticdl-test-mnist-master     1/1     Running   0          33s
elasticdl-test-mnist-ps-0       1/1     Running   0          30s
elasticdl-test-mnist-worker-0   1/1     Running   0          30s
```

We can also trace the training progress by watching the log from the master
container. The following command watches the evaluation metrics changing over
iterations.

```bash
kubectl logs elasticdl-test-mnist-master | grep "Evaluation"
```

The output looks like the following.

```txt
[2020-04-14 02:46:21,836] [INFO] [master.py:192:prepare] Evaluation service started
[2020-04-14 02:46:40,750] [INFO] [evaluation_service.py:214:complete_task] Evaluation metrics[v=50]: {'accuracy': 0.21933334}
[2020-04-14 02:46:53,827] [INFO] [evaluation_service.py:214:complete_task] Evaluation metrics[v=100]: {'accuracy': 0.5173333}
[2020-04-14 02:47:07,529] [INFO] [evaluation_service.py:214:complete_task] Evaluation metrics[v=150]: {'accuracy': 0.6253333}
[2020-04-14 02:47:23,251] [INFO] [evaluation_service.py:214:complete_task] Evaluation metrics[v=200]: {'accuracy': 0.752}
[2020-04-14 02:47:35,746] [INFO] [evaluation_service.py:214:complete_task] Evaluation metrics[v=250]: {'accuracy': 0.77}
[2020-04-14 02:47:52,082] [INFO] [master.py:249:_stop] Evaluation service stopped
```

The logs show that the accuracy reaches to 0.77 after 250 steps iteration.

## Submit the Training Job Using AllReduce

```bash
elasticdl train \
  --image_name=elasticdl:mnist_allreduce \
  --model_zoo=model_zoo \
  --model_def=mnist.mnist_functional_api.custom_model \
  --training_data=/data/mnist/train \
  --num_epochs=1 \
  --master_resource_request="cpu=0.2,memory=1024Mi" \
  --master_resource_limit="cpu=1,memory=2048Mi" \
  --worker_resource_request="cpu=0.4,memory=1024Mi" \
  --worker_resource_limit="cpu=1,memory=2048Mi" \
  --minibatch_size=64 \
  --num_minibatches_per_task=2 \
  --num_workers=2 \
  --job_name=test-mnist-allreduce \
  --image_pull_policy=Never \
  --volume="host_path=/data,mount_path=/data" \
  --distribution_strategy=AllreduceStrategy
```

After the job submission, we can run the command `kubectl get pods` to list
related containers.

```bash
NAME                                      READY   STATUS    RESTARTS   AGE
elasticdl-test-mnist-allreduce-master     1/1     Running   0          102s
elasticdl-test-mnist-allreduce-worker-0   1/1     Running   0          98s
elasticdl-test-mnist-allreduce-worker-1   1/1     Running   0          98s
```

Then, we can view the loss in the worker log using the following command

```bash
kubectl logs elasticdl-test-mnist-allreduce-worker-0 | grep Loss
```

The outputs look like.

```txt
[2020-08-27 13:22:47,930] [INFO] [worker.py:627:_process_minibatch] Loss = 2.686038017272949, steps = 2
[2020-08-27 13:23:17,254] [INFO] [worker.py:627:_process_minibatch] Loss = 0.08301685750484467, steps = 100
[2020-08-27 13:23:47,887] [INFO] [worker.py:627:_process_minibatch] Loss = 0.0823458805680275, steps = 200
[2020-08-27 13:24:19,067] [INFO] [worker.py:627:_process_minibatch] Loss = 0.14079990983009338, steps = 300
```
