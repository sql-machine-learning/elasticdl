# ElasticDL on local

This document aims to give a simple example to show how to sumbmit deep learning jobs to a local kubernetes cluster in a local computer. It helps to understand the working process of ElasticDL.


## Environment prepare

Here, we use macbook as our expriment environment.


1. Install docker

```bash
brew cask install docker
```

2. Install kunernetes

```bash
brew install kubectl
brew cask install minikube
brew install docker-machine-driver-hyperkit
```


And change permission of hyperkit:

```bash
sudo chown root:wheel /usr/local/opt/docker-machine-driver-hyperkit/bin/docker-machine-driver-hyperkit
sudo chmod u+s /usr/local/opt/docker-machine-driver-hyperkit/bin/docker-machine-driver-hyperkit
```

## Write model file

We use Tensorflow Keras API to build our models. For details, please refer to [ModelBuilding](https://github.com/sql-machine-learning/elasticdl/blob/develop/elasticdl/doc/model_building.md) part.


## Summit Job to minikube

### Install ElasticDL

```bash
git clone https://github.com/sql-machine-learning/elasticdl.git
cd elasticdl
python setup.py install
```

### Setup kubernetes related environment

```bash
kubectl apply -f manifests/examples/elasticdl-rbac.yaml
minikube start --vm-driver=hyperkit --cpus 2 --memory 6144
eval $(minikube docker-env)
bash elasticdl/docker/build_all.sh
```

### Summit a training job


You have to set right docker environment. Please run `minikube docker-env` to get docker host url and docker cert path.

A possible example could be:

```
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
  --training_data_dir=/data/mnist/train \
  --evaluation_data_dir=/data/mnist/test \
  --num_epochs=2 \
  --master_resource_request="cpu=400m,memory=1024Mi" \
  --master_resource_limit="cpu=1,memory=2048Mi" \
  --worker_resource_request="cpu=400m,memory=2048Mi" \
  --worker_resource_limit="cpu=1,memory=3072Mi" \
  --minibatch_size=64 \
  --records_per_task=100 \
  --num_workers=2 \
  --checkpoint_steps=10 \
  --evaluation_steps=15 \
  --grads_to_wait=2 \
  --job_name=test-mnist \
  --log_level=INFO \
  --image_pull_policy=Never \
  --output=model_output
```




