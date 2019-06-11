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
python elasticdl/python/elasticdl/client/client.py \
    --job_type=training \
    --job_name=local-job \
    --model_file=elasticdl/python/examples/mnist_functional_api.py \
    --training_data_dir=/data/mnist/train \
    --evaluation_data_dir=/data/mnist/test \
    --num_epochs=1 \
    --minibatch_size=64 \
    --records_per_task=100 \
    --num_workers=1 \
    --master_cpu_request=1000m \
    --master_cpu_limit=1000m \
    --master_memory_request=512Mi \
    --master_memory_limit=512Mi \
    --worker_cpu_request=1000m \
    --worker_cpu_limit=1000m \
    --worker_memory_request=1024Mi \
    --worker_memory_limit=1024Mi \
    --grads_to_wait=1 \
    --codec_type=bytes \
    --image_base=gcr.io/elasticdl/elasticdl:dev \
    --image_pull_policy=Always \
    --checkpoint_dir=/tmp \
    --checkpoint_step=10 \
    --keep_checkpoint_max=1 \
    --log_level=INFO
```

### Submit to a GKE cluster

```bash
python elasticdl/python/elasticdl/client/client.py \
    --job_type=training \
    --job_name=high-prio-job \
    --model_file=elasticdl/python/examples/mnist_functional_api.py \
    --training_data_dir=/data/mnist_nfs/mnist/train \
    --evaluation_data_dir=/data/mnist_nfs/mnist/test \
    --num_epochs=1 \
    --minibatch_size=64 \
    --records_per_task=100 \
    --num_workers=1 \
    --master_cpu_request=1000m \
    --master_cpu_limit=1000m \
    --master_memory_request=1024Mi \
    --master_memory_limit=1024Mi \
    --worker_cpu_request=3000m \
    --worker_cpu_limit=3000m \
    --worker_memory_request=4096Mi \
    --worker_memory_limit=4096Mi \
    --grads_to_wait=1 \
    --codec_type=bytes \
    --mount_path=/data \
    --volume_name=data-volume \
    --repository=gcr.io \
    --image_base=gcr.io/elasticdl/elasticdl:dev \
    --image_pull_policy=Always \
    --checkpoint_dir=/data/cp \
    --checkpoint_step=10 \
    --keep_checkpoint_max=1 \
    --log_level=INFO
```
The difference is the additional `repository` argument that points to the Docker hub used by GKE.

## Check the pod status

```bash
kubectl get pods
kubectl logs ${pod_name}
```

## Evaluate the trained model

After the training job finished, the trained model will be exported to the directory specified by parameter `--checkpoint_dir`, and ElasticDL provide user a command to reevaluate the trained model with the specified dataset using the command below to see the loss and accuracy of the trainded model:

```bash
python elasticdl/python/elasticdl/client/client.py \
    --job_type=evaluation \
    --job_name=eval-job \
    --model_file=elasticdl/python/examples/mnist_functional_api.py \
    --trained_model=/data/cp/model_${VERSION}.chkpt \
    --data_dir=/data/mnist_nfs/mnist/train \
    --minibatch_size=64 \
    --eval_cpu_request=1000m \
    --eval_cpu_limit=1000m \
    --eval_memory_request=1024Mi \
    --eval_memory_limit=1024Mi \
    --codec_type=bytes \
    --repository=gcr.io \
    --mount_path=/data \
    --volume_name=data-volume \
    --image_base=gcr.io/elasticdl/elasticdl:dev \
    --image_pull_policy=Always \
    --log_level=INFO
```
