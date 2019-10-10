# ElasticDL on Local Environment

This document aims to give a simple example to show how to submit deep learning jobs to a local kubernetes cluster in a local computer. It helps to understand the working process of ElasticDL.


## Environment preparation

Here we should install Minikube first. Please refer to the official [installation guide](https://kubernetes.io/docs/tasks/tools/install-minikube/).

In this tutorial, we use [hyperkit](https://github.com/moby/hyperkit) as the hypervisor of Minikube.

## Write model file

We use TensorFlow Keras API to build our models. Please refer to this [tutorials](model_building.md) on model building for details.

## Summit Job to Minikube

### Install ElasticDL

```bash
git clone https://github.com/sql-machine-learning/elasticdl.git
cd elasticdl
python setup.py install
```

### Setup kubernetes related environment

```bash
minikube start --vm-driver=hyperkit --cpus 2 --memory 6144
kubectl apply -f elasticdl/manifests/examples/elasticdl-rbac.yaml
eval $(minikube docker-env)
bash elasticdl/docker/build_all.sh
```

### Summit a training job


There are other docker settings that you might also want to configure prior to submitting the training job. Please run `minikube docker-env` to get docker host url and docker cert path.

A possible example could be:

```bash
export DOCKER_BASE_URL=tcp://192.168.64.5:2376
export DOCKER_TLSCERT=${HOME}/.minikube/certs/cert.pem
export DOCKER_TLSKEY=${HOME}/.minikube/certs/key.pem
```


```bash
elasticdl train \
  --image_base=elasticdl:ci \
  --docker_base_url=${DOCKER_BASE_URL} \
  --docker_tlscert=${DOCKER_TLSCERT} \
  --docker_tlskey=${DOCKER_TLSKEY} \
  --model_zoo=model_zoo \
  --model_def=mnist_functional_api.mnist_functional_api.custom_model \
  --training_data=/data/mnist/train \
  --evaluation_data=/data/mnist/test \
  --num_epochs=2 \
  --master_resource_request="cpu=400m,memory=1024Mi" \
  --master_resource_limit="cpu=1,memory=2048Mi" \
  --worker_resource_request="cpu=400m,memory=2048Mi" \
  --worker_resource_limit="cpu=1,memory=3072Mi" \
  --minibatch_size=64 \
  --num_minibatches_per_task=2 \
  --num_workers=2 \
  --checkpoint_steps=10 \
  --evaluation_steps=15 \
  --grads_to_wait=2 \
  --job_name=test-mnist \
  --log_level=INFO \
  --image_pull_policy=Never \
  --output=model_output
```


### Check job status

After submitting the job to Minikube, you can run following command to check the status of each pod:

```bash
kubectl get pods
```

You should see information on each pod like the following:

```
$kubectl get pods
NAME                            READY   STATUS    RESTARTS   AGE
elasticdl-test-mnist-master     1/1     Running   0          14s
elasticdl-test-mnist-worker-0   1/1     Running   0          11s
elasticdl-test-mnist-worker-1   1/1     Running   0          11s
```