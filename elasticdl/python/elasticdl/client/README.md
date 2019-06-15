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

There are several Keras examples provided in `elasticdl/examples` directory.

## Submit ElasticDL job

Use ElasticDL client to launch ElasticDL system on a Kubernetes cluster and submit a model, e.g. `elasticdl/examples/mnist_subclass.py` to it.

### Submit to local Kubernetes on Your Machine

```bash
python elasticdl/python/elasticdl/client/client.py train \
    --model_file=elasticdl/python/examples/mnist_functional_api.py \
    --training_data_dir=/data/mnist/train \
    --evaluation_data_dir=/data/mnist/test \
    --num_epochs=1 \
    --master_resource_request="cpu=1,memory=512Mi" \
    --master_resource_limit="cpu=1,memory=512Mi" \
    --worker_resource_request="cpu=1,memory=1024Mi" \
    --worker_resource_limit="cpu=1,memory=1024Mi" \
    --minibatch_size=10 \
    --records_per_task=100 \
    --num_workers=1 \
    --grads_to_wait=2 \
    --codec_type=tf_example \
    --job_name=test \
    --image_base=elasticdl:dev \
    --log_level=INFO
```

### Submit to a GKE cluster

```bash
python elasticdl/python/elasticdl/client/client.py train \
    --job_name=test \
    --model_file=elasticdl/python/examples/mnist_functional_api.py \
    --training_data_dir=/data/mnist_nfs/train \
    --evaluation_data_dir=/data/mnist_nfs/test \
    --num_epochs=1 \
    --minibatch_size=10 \
    --records_per_task=100 \
    --num_workers=1 \
    --master_pod_priority=highest-priority \
    --worker_pod_priority=high-priority \
    --master_resource_request="cpu=1,memory=2048Mi" \
    --master_resource_limit="cpu=1,memory=2048Mi" \
    --worker_resource_request="cpu=2,memory=4096Mi" \
    --worker_resource_limit="cpu=2,memory=4096Mi" \
    --grads_to_wait=2 \
    --codec_type=tf_example \
    --mount_path=/data \
    --volume_name=data-volume \
    --image_base=gcr.io/elasticdl/mnist:dev \
    --image_pull_policy=Always \
    --log_level=INFO \
    --push_image
```
The difference is that we need to push the built image to a remote image registry used by GKE.

## Check the pod status

```bash
kubectl get pods
kubectl logs ${pod_name}
```
