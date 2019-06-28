# ElasticDL Client: Submit ElasticDL Job to Kubernetes 

Currently for Mac docker-for-desktop only.

## Check Environment

Make sure the Kubernetes docker-for-desktop (not minikube) is installed on your Mac.


## Write a Keras Model

**(TODO: Describe programming API)**

There are several Keras examples provided in `elasticdl/examples` directory.

## Submit ElasticDL Job In Development Mode

### Download ElasticDL Source Code
```bash
git clone https://github.com/wangkuiyi/elasticdl.git
cd elasticdl
```

Use ElasticDL client to launch ElasticDL system on a Kubernetes cluster and submit a model, e.g. `/Users/${USER_NAME}/elasticdl/elasticdl/examples/mnist_subclass.py` to it.

### Submit to local Kubernetes on Your Machine

For demonstration purposes, we use the data stored on `elasticdl:ci` Docker image.

First we build all development Docker images, which include `elasticdl:ci` image:

```bash
elasticdl/docker/build_all.sh
```

Submit training job:

```bash
python -m elasticdl.python.client.client train \
    --image_base=elasticdl:ci \
    --model_file=$(pwd)/elasticdl/python/examples/mnist_subclass.py \
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
    --checkpoint_steps=2 \
    --grads_to_wait=2 \
    --job_name=test \
    --image_name=elasticdl:dev \
    --image_pull_policy=Never \
    --log_level=INFO
```

### Submit to a GKE cluster

```bash
python -m elasticdl.python.client.client train \
    --job_name=test \
    --image_name=gcr.io/elasticdl/mnist:dev \
    --model_file=$(pwd)/elasticdl/python/examples/mnist_subclass.py \
    --training_data_dir=/data/mnist_nfs/mnist/train \
    --evaluation_data_dir=/data/mnist_nfs/mnist/test \
    --num_epochs=1 \
    --minibatch_size=10 \
    --records_per_task=100 \
    --num_workers=1 \
    --checkpoint_steps=2 \
    --master_pod_priority=high-priority \
    --worker_pod_priority=high-priority \
    --master_resource_request="cpu=1,memory=2048Mi" \
    --master_resource_limit="cpu=1,memory=2048Mi" \
    --worker_resource_request="cpu=2,memory=4096Mi" \
    --worker_resource_limit="cpu=2,memory=4096Mi" \
    --grads_to_wait=2 \
    --mount_path=/data \
    --volume_name=data-volume \
    --image_pull_policy=Always \
    --log_level=INFO \
    --push_image
```
The difference is that we need to push the built image to a remote image registry used by GKE.

## Submit ElasticDL Job In Command Line Mode

### Download ElasticDL Source Code And Build Wheel Package
```bash
git clone https://github.com/wangkuiyi/elasticdl.git
cd elasticdl
```

### Build And Install Wheel Package From Source Code
```bash
python3 setup.py bdist_wheel
pip install dist/ElasticDL-0.0.1-py3-none-any.whl
```

### Submit Jobs

Same as in the development mode, just replace `python -m elasticdl.python.client.client` part with `elasticdl`.


## Check the pod status

```bash
kubectl get pods
kubectl logs ${pod_name}
```
