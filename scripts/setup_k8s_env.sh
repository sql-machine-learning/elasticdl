#!/bin/bash

set -e

export MINIKUBE_WANTUPDATENOTIFICATION=false
export MINIKUBE_WANTREPORTERRORPROMPT=false
export MINIKUBE_HOME=$HOME
export CHANGE_MINIKUBE_NONE_USER=true
export KUBECONFIG=$HOME/.kube/config
export K8S_VERSION=1.14.0
export MINIKUBE_VERSION=1.1.1

curl -Lo kubectl https://storage.googleapis.com/kubernetes-release/release/v${K8S_VERSION}/bin/linux/amd64/kubectl && chmod +x kubectl && sudo mv kubectl /usr/local/bin/
curl -Lo minikube https://storage.googleapis.com/minikube/releases/v${MINIKUBE_VERSION}/minikube-linux-amd64 && chmod +x minikube && sudo mv minikube /usr/local/bin/
mkdir -p $HOME/.kube $HOME/.minikube
touch ${KUBECONFIG}
sudo minikube start --vm-driver=none --kubernetes-version=v${K8S_VERSION} --cpus 2 --memory 6144
sudo chown -R travis: $HOME/.minikube/
kubectl cluster-info
