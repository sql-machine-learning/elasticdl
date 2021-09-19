# Train PyTorch Models using ElasticDL on Personal Computer

This document shows how to run an ElasticDL AllReduce job to train a PyTorch
model using MNIST dataset on Minikube.

## Prerequisites

1. Install Minikube, preferably >= v1.11.0, following the installation
   [guide](https://kubernetes.io/docs/tasks/tools/install-minikube).  Minikube
   runs a single-node Kubernetes cluster in a virtual machine on your personal
   computer.

1. Install Docker CE, preferably >= 18.x, following the
   [guide](https://docs.docker.com/docker-for-mac/install/) for building Docker
   images containing user-defined models and the ElasticDL framework.

1. Install Python, preferably >= 3.6, because the ElasticDL command-line tool is
   in Python.

## Models

In this tutorial, we use the model defined in the [model
zoo](https://github.com/sql-machine-learning/elasticdl/tree/develop/model_zoo/mnist/mnist_pytorch.py)
directory.  This model is defined using PyTorch API.

## Datasets

We use the [MINST](https://www.kaggle.com/jidhumohan/mnist-png/download)
dataset in this tutorial. After downloading the dataset, we should
unzip it into a local directory.

## The Kubernetes Cluster

The following command starts a Kubernetes cluster locally using Minikube.  It
uses [VirtualBox](https://www.virtualbox.org/), a hypervisor coming with
macOS, to create the virtual machine cluster.

```bash
minikube start --vm-driver=virtualbox \
  --cpus 2 --memory 6144 --disk-size=50gb 
eval $(minikube docker-env)
```

The command `minikube docker-env` returns a set of Bash environment variable
to configure your local environment to re-use the Docker daemon inside
the Minikube instance.

The following command is necessary to enable
[RBAC](https://kubernetes.io/docs/reference/access-authn-authz/rbac/) of
Kubernetes.

```bash
kubectl apply -f \
  https://raw.githubusercontent.com/sql-machine-learning/elasticdl/develop/elasticdl/manifests/elasticdl-rbac.yaml
```

If you happen to live in a region where `raw.githubusercontent.com` is banned,
you might want to Git clone the above repository to get the YAML file.

## Install ElasticDL Client Tool

The following command installs the command line tool `elasticdl`, which talks to
the Kubernetes cluster and operates ElasticDL jobs.

```bash
pip install elasticdl_client
```

## Build the Docker Image with Model Definition

Kubernetes runs Docker containers, so we need to put user-defined models,
the ElasticDL api package and all dependencies into a Docker image.

In this tutorial, we use a predefined model in the ElasticDL repository.  To
retrieve the source code, please run the following command.

```bash
git clone https://github.com/sql-machine-learning/elasticdl
```

The model definition is in directory [elasticdl/model_zoo/mnist/mnist_pytorch.py](https://github.com/sql-machine-learning/elasticdl/blob/develop/model_zoo/mnist/mnist_pytorch.py).

We build the image based on `horovod/horovod:0.20.0-tf2.3.0-torch1.6.0-mxnet1.5.0-py3.7-cpu`
and the dockerfile is

```text
FROM horovod/horovod:0.20.0-tf2.3.0-torch1.6.0-mxnet1.5.0-py3.7-cpu as base

RUN pip install opencv-python
RUN apt update
RUN apt install -y libgl1-mesa-glx libglib2.0-dev

RUN HOROVOD_WITHOUT_MPI=1 \
    HOROVOD_WITH_GLOO=1 \
    HOROVOD_WITHOUT_MXNET=1 \
    HOROVOD_WITH_TENSORFLOW=1 \
    HOROVOD_WITH_PYTORCH=1  \
    pip install horovod==0.21.0

RUN pip install elasticdl_api

COPY ./model_zoo model_zoo
ENV PYTHONUNBUFFERED 0
```

Then, we use docker to build the image

```bash
docker build -t elasticdl:mnist_pytorch -f ${mnist_dockerfile} .
```

## Submit the Training Job

The following command submits a training job:

```bash
elasticdl train \
  --image_name=elasticdl/elasticdl:v1.0.0 \
  --worker_image=elasticdl:mnist_pytorch \
  --job_command="python -m model_zoo.mnist.mnist_pytorch --batch_size 64 --num_epochs 1 --training_data=/data/mnist_png/training --validation_data=/data/mnist_png/testing" \
  --num_minibatches_per_task=2 \
  --num_workers=2 \
  --master_resource_request="cpu=0.2,memory=1024Mi" \
  --master_resource_limit="cpu=1,memory=2048Mi" \
  --worker_resource_request="cpu=0.3,memory=1024Mi" \
  --worker_resource_limit="cpu=1,memory=2048Mi" \
  --envs="USE_TORCH=true,HOROVOD_GLOO_TIMEOUT_SECONDS=60,PYTHONUNBUFFERED=true" \
  --job_name=test-mnist-allreduce \
  --image_pull_policy=Never \
  --volume="host_path={mnist_data_dir},mount_path=/data" \
  --distribution_strategy=AllreduceStrategy 
```

`--image_name` is the image to launch the ElasticDL master which
has nothing to do with the estimator model. The ElasticDL master is
responsible for launching pod and assigning data shards to workers with
elasticity and fault-tolerance.

`{mnist_data_dir}` is the absolute path of the `./data` with the directory of
`mnist_png`. Here, the option `--volume="host_path={mnist_data_dir},mount_path=/data"`
bind mount it into the containers/pods.

The option `--num_workers=2` tells the master to start 2 worker pods.

### Check Job Status

After the job submission, we can run the command `kubectl get pods` to list
related containers.

```bash
NAME                                    READY   STATUS    RESTARTS   AGE
elasticdl-test-mnist-allreduce-master   1/1     Running   0          7s
test-mnist-allreduce-edljob-worker-0    1/1     Running   0          5s
test-mnist-allreduce-edljob-worker-1    1/1     Running   0          5s
```

## Train an PyTorch Model Using ElasticDL with Your Dataset

In order to support fault-tolerance and elasticity with ElasticDL, you only
need to create a custom dataset and wrap the function to perform forward and
backward computation using ElasticDL APIs.

### Create a Dataset With the RecordIndexService of ElasticDL

ElasticDL can split the total dataset into shards and assign those
shards to workers. If some workers fail, ElasticDL can re-assign
shards of failed workers to other running workers. We can get sample
indices in those shards by `RecordIndexService`. We can create a
dataset to read images by indices from the `RecordIndexService`.

```python
class ElasticDataset(Dataset):
    def __init__(self, images, data_shard_service=None):
        """The dataset supports elastic training.

        Args:
            images: A list with tuples like (image_path, label_index).
            For example, we can use `torchvision.datasets.ImageFolder`
            to get the list.
            data_shard_service: If we want to use elastic training, we
            need to use the `data_shard_service` of the elastic controller
            in elasticai_api.
        """
        self.data_shard_service = data_shard_service
        self._images = images

    def __len__(self):
        if self.data_shard_service:
            # Set the maxsize because the size of dataset is not fixed
            # when using dynamic sharding
            return sys.maxsize
        else:
            return len(self._images)

    def __getitem__(self, index):
        if self.data_shard_service:
            index = self.data_shard_service.fetch_record_index()
            return self.read_image(index)
        else:
            return self.read_image(index)

    def read_image(self, index):
        image_path, label = self._images[index]
        image = cv2.imread(image_path)
        image = np.array(image / 255.0, np.float32)
        image = image.reshape(3, 28, 28)
        return image, label


if __name__ == "__main__":
    ...
    data_shard_service = RecordIndexService(
        batch_size=args.batch_size,
        dataset_size=len(train_data.imgs),
        num_epochs=args.num_epochs,
        shuffle=True,
        dataset_name="mnist_training_data",
    )
    train_dataset = ElasticDataset(train_data.imgs, data_shard_service)
    ...
```

### Create an ElasticDL Controller to Wrap the Forward and Backward Computation

In ElasticDL AllReduce, we need to create a `PyTorchAllReduceController`
of ElasticDL. At the begining, The controller can broadcast the model and
optimizer. If some workers fail, the controller will re-initialize Horovod
using a new world. After creating the controller, we should wrap the function
to perform the forward and backward computation by `elastic_run`.

```python
def train_one_batch(model, optimizer, data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss


if __name__ == "__main__":
    ...
    model = ...
    optimizer = ...
    allreduce_controller = PyTorchAllReduceController(data_shard_service)
    allreduce_controller.set_broadcast_model(model)
    allreduce_controller.set_broadcast_optimizer(optimizer)
    # Use the elastic function to wrap the training function with a batch.
    elastic_train_one_batch = allreduce_controller.elastic_run(train_one_batch)

    with allreduce_controller.scope():
        for batch_idx, (data, target) in enumerate(train_loader):
            loss = elastic_train_one_batch(model, optimizer, data, target)
    ...
```