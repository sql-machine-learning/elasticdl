# ElasticDL in local

This document aims to give a simple example to show how to sumbmit deep learning jobs to a local kubernetes cluster in a local computer. It helps to understand the working process of ElasticDL.


## Environment Prepare

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

## Write Model File

We use Tensorflow Keras API to build our models. For details, please refer to [ModelBuilding](https://github.com/sql-machine-learning/elasticdl/blob/develop/elasticdl/doc/model_building.md) part.


## Summit Job to minikube

### Install ElasticDL


### Start minikube

```bash
minikube start --vm-driver hyperkit
eval $(minikube docker-env)
```

### Build base docker images


### Summit job




