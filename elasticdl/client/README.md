# ElasticDL Client: Submit ElasticDL Job to Kubernetes 

Currently for Mac docker-for-desktop only.

## Check Environment

Make sure the Kubernetes docker-for-desktop (not minikube) is installed on your Mac.

## Download ElasticDL Source Code
```bash
git clone https://github.com/wangkuiyi/elasticdl.git
cd elasticdl
```

## Build ElasticDL Development Docker Image
```bash
docker build -t elasticdl:dev -f dockerfile/elasticdl.dev .
```
The Kubernetes example use `elasticdl:dev` Docker image as the base master/worker image.


## Write a Keras Model

**(TODO: Describe programming API)**

There are several Keras examples provided in `edl_k8s_examples` directory.

## Submit ElasticDL job

Use ElasticDL client to launch ElasticDL system on a Kubernetes cluster and submit a model, e.g. `edl_k8s_examples/mnist_model.py` to it.

### Submit to local Kubernetes on Your Machine

```bash
python elasticdl/client/client.py \
    --model_file=edl_k8s_examples/mnist_subclass.py \
    --train_data_dir=/data/mnist/train \
    --num_epoch=1 \
    --minibatch_size=10 \
    --record_per_task=100 \
    --num_worker=1 \
    --grads_to_wait=2 \
    --codec-type=tf_example \
    --job_name=test \
    --image_base=elasticdl:dev
```

### Submit to a GKE cluster

```bash
python elasticdl/client/client.py \
    --model_file=edl_k8s_examples/mnist_subclass.py \
    --train_data_dir=/data/mnist/train \
    --num_epoch=1 \
    --minibatch_size=10 \
    --record_per_task=100 \
    --num_worker=1 \
    --grads_to_wait=2 \
    --codec-type=tf_example \
    --job_name=test \
    --repository=gcr.io \
    --image_base=gcr.io/elasticdl/mnist:dev
```
The difference is the additional `repository` argument that points to the Docker hub used by GKE.

## Check the pod status

```bash
kubectl get pods
kubectl logs ${pod_name}
```
