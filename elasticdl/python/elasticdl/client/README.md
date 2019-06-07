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
    --model_file=elasticdl/python/examples/mnist_functional_api.py \
    --training_data_dir=/data/mnist/train \
    --evaluation_data_dir=/data/mnist/test \
    --num_epoch=1 \
    --master_cpu_request=1000m \
    --master_cpu_limit=1000m \
    --master_memory_request=512Mi \
    --master_memory_limit=512Mi \
    --worker_cpu_request=1000m \
    --worker_cpu_limit=1000m \
    --worker_memory_request=1024Mi \
    --worker_memory_limit=1024Mi \
    --minibatch_size=10 \
    --record_per_task=100 \
    --num_worker=1 \
    --grads_to_wait=2 \
    --codec_type=tf_example \
    --job_name=test \
    --image_base=elasticdl:dev \
    --log_level=INFO
```

### Submit to a GKE cluster

```bash
python elasticdl/python/elasticdl/client/client.py \
    --job_name=test \
    --model_file=elasticdl/python/examples/mnist_functional_api.py \
    --training_data_dir=/data/mnist_nfs/train \
    --evaluation_data_dir=/data/mnist_nfs/test \
    --num_epoch=1 \
    --minibatch_size=10 \
    --record_per_task=100 \
    --num_worker=1 \
    --master_pod_priority=highest-priority \
    --worker_pod_priority=high-priority \
    --master_cpu_request=1000m \
    --master_cpu_limit=1000m \
    --master_memory_request=2048Mi \
    --master_memory_limit=2048Mi \
    --worker_cpu_request=2000m \
    --worker_cpu_limit=2000m \
    --worker_memory_request=4096Mi \
    --worker_memory_limit=4096Mi \
    --grads_to_wait=2 \
    --codec_type=tf_example \
    --mount_path=/data \
    --volume_name=data-volume \
    --repository=gcr.io \
    --image_base=gcr.io/elasticdl/mnist:dev \
    --image_pull_policy=Always \
    --log_level_INFO
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
python3 elasticdl/python/elasticdl/client/client.py \
    --job_type=evaluation \
    --job_name=high-eval-job \
    --model_file=elasticdl/python/examples/mnist_subclass.py \
    --trained_model=/data/cp/model_${version}.chkpt \
    --data_dir=/data/mnist_nfs/mnist/train \
    --pod_priority=high-priority \
    --cpu_request=1000m \
    --cpu_limit=1000m \
    --memory_request=1024Mi \
    --memory_limit=1024Mi \
    --codec_type=bytes \
    --repository=gcr.io \
    --mount_path=/data \
    --volume_name=data-volume \
    --image_base=gcr.io/elasticdl/elasticdl:dev \
    --image_pull_policy=Always \
    --log_level=INFO
```
