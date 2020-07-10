# ElasticDL on Personal Computer

This document shows how to run ElasticDL jobs on your personal computer using
Minikube.

## Prerequisites

1. Install Minikube, preferably >= v1.11.0, following the [installation
   guide](https://kubernetes.io/docs/tasks/tools/install-minikube).  Minikube
   runs a single-node Kubernetes cluster in a virtual machine on your personal
   computer.

Minikube works with a variety of virtual machine hypervisors.  In this tutorial,
we use [hyperkit](https://github.com/moby/hyperkit) that comes with macOS. If
you want to use other hypervisors like VirtualBox, please install them
accordingly.

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

After the running of this command, we will see dataset files in the current
directory.

## The Kubernetes Cluster

The following command to start a Kubernetes cluster locally using Minikube.

```bash
minikube start --vm-driver=hyperkit \
  --cpus 2 --memory 6144 --disk-size=50gb \
  --mount=true --mount-string="./data:/data"
eval $(minikube docker-env)
```

The above command-line option `--mount-string` exposes the directory `./data` on
the host to Minikube, so we can bind mount it into containers running on the
local Kubernetes cluster.

## Install ElasticDL Client Tool

```bash
pip install elasticdl_client
```

## Build the Docker Image

Kubernetes runs Docker containers, so we need to release the training program,
composed of user-defined models and ElasticDL, into a Docker image.

In this tutorial, we use a predefined model in the ElasticDL repository. The
following command retrieves the source code of the user-defined model into
`./elasticdl/model_zoo`.

```bash
git clone https://github.com/sql-machine-learning/elasticdl.git
```

The following commands build the Docker image `elasticdl:mnist`. Please feel
free to name it in any other name you like.

```bash
cd elasticdl/model_zoo
elasticdl zoo init
elasticdl zoo build --image=elasticdl:mnist .
```

## Authorize the Job Execution

If you are going to run ElasticDL job in Minikube for the first time. Please
execute the following command to authorize the execution. As ElasticDL is a
Kubernetes-native deep learning framework, we execute it to authorize the
master pod to create and monitor the worker/ps pods. The command need to be
executed only once.

```bash
kubectl apply -f ../elasticdl/manifests/elasticdl-rbac.yaml
```

## Submit the Training Job

The following command submits a training job:

```bash
elasticdl train \
  --image_name=elasticdl:mnist \
  --model_zoo=model_zoo \
  --model_def=mnist_functional_api.mnist_functional_api.custom_model \
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
  --grads_to_wait=1 \
  --job_name=test-mnist \
  --log_level=INFO \
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

## Check Job Status

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
