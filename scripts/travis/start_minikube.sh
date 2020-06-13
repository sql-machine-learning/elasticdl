#!/bin/bash
# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

# install Kuberneters Python client to validate job status
sudo pip3 install kubernetes

export MINIKUBE_WANTUPDATENOTIFICATION=false
export MINIKUBE_WANTREPORTERRORPROMPT=false
export MINIKUBE_HOME=$HOME
export KUBECONFIG=$HOME/.kube/config
export K8S_VERSION=v1.18.3
export MINIKUBE_VERSION=v1.11.0

# Download and install kubectl
KUBECTL_BUCKET=https://storage.googleapis.com/kubernetes-release
curl -Lo kubectl "$KUBECTL_BUCKET/release/$K8S_VERSION/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Download and install minikube
MINIKUBE_BUCKET=https://storage.googleapis.com/minikube/releases
curl -Lo minikube "$MINIKUBE_BUCKET/$MINIKUBE_VERSION/minikube-linux-amd64"
chmod +x minikube
sudo mv minikube /usr/local/bin/

mkdir -p "$HOME"/.kube "$HOME"/.minikube
touch "$KUBECONFIG"
minikube start --vm-driver=docker --kubernetes-version="$K8S_VERSION"
kubectl cluster-info
