# Distributed Training of DeepCTR Estimator using ElasticDL on Kubernetes

This document shows how to run a distributed training job of a deepctr
estimator model (DeepFM) using [ElasticDL](https://github.com/sql-machine-learning/elasticdl)
on Kubernetes.

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

In this tutorial, we use a [DeepFM estimator](https://github.com/shenweichen/DeepCTR/blob/master/deepctr/estimator/models/deepfm.py)
model in DeepCTR. The complete program to train the model with the
dataset definition is in [ElasticDL model zoo](https://github.com/sql-machine-learning/elasticdl/tree/develop/model_zoo/deepctr).

## Dataset

In this tutorial, We use the [criteo dataset](https://github.com/shenweichen/DeepCTR/blob/master/examples/criteo_sample.txt)
in DeepCTR examples.

```bash
mkdir ./data
wget https://github.com/shenweichen/DeepCTR/blob/master/examples/criteo_sample.txt -O ./data/criteo_sample.txt
```

## The Kubernetes Cluster

The following command starts a Kubernetes cluster locally using Minikube.  It
uses [VirtualBox](https://www.virtualbox.org/), a hypervisor coming with
macOS, to create the virtual machine cluster.

```bash
minikube start --vm-driver=virtualbox \
  --cpus 2 --memory 6144 --disk-size=50gb 
eval $(minikube docker-env)
```

The command `minikube docker-env` returns a set of Bash environment variable
to configure your local environment to re-use the Docker daemon inside
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

Kubernetes runs Docker containers, so we need to put user-defined models,
the ElasticDL api package and all dependencies into a Docker image.

In this tutorial, we use a complete program using a DeepFM estimator model of DeepCTR
in the ElasticDL repository. To retrieve the source code, please run the following command.

```bash
git clone https://github.com/sql-machine-learning/elasticdl
```

Complete codes are in directory [elasticdl/model_zoo/deepctr](https://github.com/sql-machine-learning/elasticdl/tree/develop/model_zoo/deepctr).

We build the image based on tensorflow:1.13.2 and the dockerfile
is

```text
FROM tensorflow/tensorflow:1.13.2-py3 as base

RUN pip install elasticdl_api
RUN pip install deepctr

COPY ./model_zoo model_zoo
```

Then, we use docker to build the image

```bash
docker build -t elasticdl:deepctr_estimator -f ${deepctr_dockerfile} .
```

## Submit the Training Job

The following command submits a training job:

```bash
elasticdl train \
  --image_name=elasticdl/elasticdl:1.0.0 \
  --worker_image=elasticdl:deepctr_estimator \
  --ps_image=elasticdl:deepctr_estimator \
  --job_command="python -m model_zoo.deepctr.deepfm_estimator --training_data=/data/criteo_sample.txt --validation_data=/data/criteo_sample.txt" \
  --num_workers=1 \
  --num_ps=1 \
  --num_evaluator=1 \
  --master_resource_request="cpu=0.2,memory=1024Mi" \
  --master_resource_limit="cpu=1,memory=2048Mi" \
  --ps_resource_request="cpu=0.2,memory=1024Mi" \
  --ps_resource_limit="cpu=1,memory=2048Mi" \
  --worker_resource_request="cpu=0.3,memory=1024Mi" \
  --worker_resource_limit="cpu=1,memory=2048Mi" \
  --chief_resource_request="cpu=0.3,memory=1024Mi" \
  --chief_resource_limit="cpu=1,memory=2048Mi" \
  --evaluator_resource_request="cpu=0.3,memory=1024Mi" \
  --evaluator_resource_limit="cpu=1,memory=2048Mi" \
  --job_name=test-deepfm-estimator \
  --distribution_strategy=ParameterServerStrategy \
  --need_tf_config=true \
  --volume="host_path={criteo_data_path},mount_path=/data" \
```

`--image_name` is the image to launch the ElasticDL master which
has nothing to do with the estimator model. The ElasticDL master is
responsible for launching pod and assigning data shards to workers with
elasticity and fault-tolerance.

`{criteo_data_path}` is the absolute path of the `./data` with `criteo_sample.txt`.
Here, the option `--volume="host_path={criteo_data_path},mount_path=/data"`
bind mount it into the containers/pods.

The option `--num_workers=1` tells the master to start a worker pod.
The option `--num_ps=1` tells the master to start a ps pod.
The option `--num_evaluator=1` tells the master to start an evaluator pod.

And the master will start a chief worker for a TensorFlow estimator model by default.

### Check Job Status

After the job submission, we can run the command `kubectl get pods` to list
related containers.

```bash
NAME                                     READY   STATUS    RESTARTS   AGE
elasticdl-test-deepctr-estimator-master     1/1     Running   0          9s
test-deepctr-estimator-edljob-chief-0       1/1     Running   0          6s
test-deepctr-estimator-edljob-evaluator-0   0/1     Pending   0          6s
test-deepctr-estimator-edljob-ps-0          1/1     Running   0          7s
test-deepctr-estimator-edljob-worker-0      1/1     Running   0          6s
```

We can view the log of workers by `kubectl logs test-deepctr-estimator-edljob-chief-0`.

```text
INFO:tensorflow:global_step/sec: 4.84156
INFO:tensorflow:global_step/sec: 4.84156
INFO:tensorflow:Saving checkpoints for 203 into /data/ckpts/model.ckpt.
INFO:tensorflow:Saving checkpoints for 203 into /data/ckpts/model.ckpt.
INFO:tensorflow:global_step/sec: 7.05433
INFO:tensorflow:global_step/sec: 7.05433
```
