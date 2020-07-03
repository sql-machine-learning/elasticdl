# ElasticDL on Local Environment

This document aims to give a simple example to show how to submit deep learning
jobs to a local kubernetes cluster in a local computer. It helps to understand
the working process of ElasticDL.

## Environment preparation

1. Install Minikube >= v1.11.0. Please refer to the official
[installation guide](https://kubernetes.io/docs/tasks/tools/install-minikube/).
In this tutorial, we use [hyperkit](https://github.com/moby/hyperkit) as the
hypervisor of Minikube.
1. Install [Docker CE >= 18.x](https://docs.docker.com/docker-for-mac/install/)
for building the Docker images of the distributed ElasticDL jobs.
1. Install Python >= 3.6.

## Write model file

We use TensorFlow Keras API to build our models. Please refer to this
[tutorials](model_building.md) on model building for details.  In this tutorial,
we use a model predefined in model zoo directory.

## Summit Job to Minikube

### Install ElasticDL Client

```bash
pip install elasticdl_client
```

Clone elasticdl repo for model zoo and some scripts.

```bash
git clone https://github.com/sql-machine-learning/elasticdl.git
```

### Start Kubernetes Cluster

We start minikube with a command-line option `--mount-string`, which mounts the host directory `$DATA_PATH` to `/data` path in all minikube containers.

```bash
export DATA_PATH={a_folder_path_to_store_training_data}
minikube start --vm-driver=hyperkit --cpus 2 --memory 6144 --disk-size=50gb --mount=true --mount-string="$DATA_PATH:/data"
cd elasticdl
kubectl apply -f elasticdl/manifests/elasticdl-rbac.yaml
eval $(minikube docker-env)
```

### Build the Docker image for distributed training

```bash
cd model_zoo
elasticdl zoo init
elasticdl zoo build --image=elasticdl:mnist .
```

We use the model predefined in model zoo directory. The model definition will
be packed into the new Docker image `elasticdl:mnist`.

### Prepare the dataset

We generate MNIST training and evaluation data in RecordIO format. We provide a
script in elasticdl repo.

```bash
docker pull elasticdl/elasticdl:dev
cd {elasticdl_repo_root}
docker run --rm -it \
  -v $HOME/.keras/datasets:/root/.keras/datasets \
  -v $PWD:/work \
  -w /work elasticdl/elasticdl:dev \
  bash -c "scripts/gen_dataset.sh data"
cp -r data/* $DATA_PATH
```

We generate datasets and copy them to `$DATA_PATH`.

### Summit a training job

We use the following command to submit a training job:

```bash
elasticdl train \
  --image_name=elasticdl:mnist \
  --model_zoo=model_zoo \
  --model_def=mnist_functional_api.mnist_functional_api.custom_model \
  --training_data=/data/mnist/train \
  --validation_data=/data/mnist/test \
  --num_epochs=2 \
  --master_resource_request="cpu=0.2,memory=1024Mi" \
  --master_resource_limit="cpu=1,memory=2048Mi" \
  --worker_resource_request="cpu=0.4,memory=1024Mi" \
  --worker_resource_limit="cpu=1,memory=2048Mi" \
  --ps_resource_request="cpu=0.2,memory=1024Mi" \
  --ps_resource_limit="cpu=1,memory=2048Mi" \
  --minibatch_size=64 \
  --num_minibatches_per_task=2 \
  --num_ps_pods=1 \
  --num_workers=1 \
  --evaluation_steps=50 \
  --grads_to_wait=1 \
  --job_name=test-mnist \
  --log_level=INFO \
  --image_pull_policy=Never \
  --volume="/data,mount_path=/data" \
  --distribution_strategy=ParameterServerStrategy
```

`image_name` is the Docker image name for the distributed ElasticDL job. We built
it using the `elasticdl zoo build` command above.

The directory to store the training and validation data are mounted into Minikube
in the previous step. We will then mount it in the path `/data` inside the pod.

In this example, we use parameter server strategy. We launch a master pod, a
parameter server(PS) pod and a worker pod. The worker pod gets model parameters
from the PS pod, computes gradients and sends computed gradients to the PS
pod. The PS pod iteratively updates these model parameters using gradients sent
by the worker pod. For more details about parameter server strategy, please
refer to the [design
doc](https://github.com/sql-machine-learning/elasticdl/blob/develop/docs/designs/parameter_server.md).

### Check job status

After submitting the job to Minikube, we can run the following command to check
the status of each pod:

```bash
kubectl get pods
```

We will see the status of each pod:

```bash
NAME                            READY   STATUS    RESTARTS   AGE
elasticdl-test-mnist-master     1/1     Running   0          33s
elasticdl-test-mnist-ps-0       1/1     Running   0          30s
elasticdl-test-mnist-worker-0   1/1     Running   0          30s
```

To see the loss in the worker pod:

```bash
kubectl logs elasticdl-test-mnist-worker-0 | grep "Loss"
```

We will see following logs:

```txt
[2020-04-14 02:46:28,535] [INFO] [worker.py:879:_process_minibatch] Loss is 3.07190203666687
[2020-04-14 02:46:28,920] [INFO] [worker.py:879:_process_minibatch] Loss is 9.413976669311523
[2020-04-14 02:46:29,120] [INFO] [worker.py:879:_process_minibatch] Loss is 3.9641590118408203
[2020-04-14 02:46:29,344] [INFO] [worker.py:879:_process_minibatch] Loss is 15.329755783081055
[2020-04-14 02:46:29,551] [INFO] [worker.py:879:_process_minibatch] Loss is 3.8414430618286133
[2020-04-14 02:46:29,817] [INFO] [worker.py:879:_process_minibatch] Loss is 2.7703640460968018
[2020-04-14 02:46:30,041] [INFO] [worker.py:879:_process_minibatch] Loss is 6.920175075531006
[2020-04-14 02:46:30,242] [INFO] [worker.py:879:_process_minibatch] Loss is 4.375149250030518
[2020-04-14 02:46:30,433] [INFO] [worker.py:879:_process_minibatch] Loss is 8.31199836730957
[2020-04-14 02:46:30,650] [INFO] [worker.py:879:_process_minibatch] Loss is 5.039440155029297
[2020-04-14 02:46:30,853] [INFO] [worker.py:879:_process_minibatch] Loss is 22.80319595336914
[2020-04-14 02:46:31,132] [INFO] [worker.py:879:_process_minibatch] Loss is 4.777717590332031
[2020-04-14 02:46:31,319] [INFO] [worker.py:879:_process_minibatch] Loss is 11.329744338989258
[2020-04-14 02:46:31,529] [INFO] [worker.py:879:_process_minibatch] Loss is 7.414562225341797
[2020-04-14 02:46:31,733] [INFO] [worker.py:879:_process_minibatch] Loss is 6.1839070320129395
[2020-04-14 02:46:31,932] [INFO] [worker.py:879:_process_minibatch] Loss is 4.577566146850586
[2020-04-14 02:46:32,125] [INFO] [worker.py:879:_process_minibatch] Loss is 4.547096252441406
[2020-04-14 02:46:32,326] [INFO] [worker.py:879:_process_minibatch] Loss is 6.603780269622803
[2020-04-14 02:46:32,510] [INFO] [worker.py:879:_process_minibatch] Loss is 2.7861897945404053
[2020-04-14 02:46:32,720] [INFO] [worker.py:879:_process_minibatch] Loss is 1.568850040435791
[2020-04-14 02:46:32,925] [INFO] [worker.py:879:_process_minibatch] Loss is 1.0977835655212402
[2020-04-14 02:46:33,123] [INFO] [worker.py:879:_process_minibatch] Loss is 0.8362151384353638
[2020-04-14 02:46:33,315] [INFO] [worker.py:879:_process_minibatch] Loss is 1.146580696105957
[2020-04-14 02:46:33,518] [INFO] [worker.py:879:_process_minibatch] Loss is 1.4624073505401611
[2020-04-14 02:46:33,648] [INFO] [worker.py:879:_process_minibatch] Loss is 0.9980261921882629
[2020-04-14 02:46:33,851] [INFO] [worker.py:879:_process_minibatch] Loss is 0.47116899490356445
[2020-04-14 02:46:34,037] [INFO] [worker.py:879:_process_minibatch] Loss is 0.9414381384849548
```

To see evaluation metrics in the master pod:

```bash
kubectl logs elasticdl-test-mnist-master | grep "Evaluation"
```

We will see following logs:

```txt
[2020-04-14 02:46:21,836] [INFO] [master.py:192:prepare] Evaluation service started
[2020-04-14 02:46:40,750] [INFO] [evaluation_service.py:214:complete_task] Evaluation metrics[v=50]: {'accuracy': 0.21933334}
[2020-04-14 02:46:53,827] [INFO] [evaluation_service.py:214:complete_task] Evaluation metrics[v=100]: {'accuracy': 0.5173333}
[2020-04-14 02:47:07,529] [INFO] [evaluation_service.py:214:complete_task] Evaluation metrics[v=150]: {'accuracy': 0.6253333}
[2020-04-14 02:47:23,251] [INFO] [evaluation_service.py:214:complete_task] Evaluation metrics[v=200]: {'accuracy': 0.752}
[2020-04-14 02:47:35,746] [INFO] [evaluation_service.py:214:complete_task] Evaluation metrics[v=250]: {'accuracy': 0.77}
[2020-04-14 02:47:52,082] [INFO] [master.py:249:_stop] Evaluation service stopped
```

The logs show that the accuracy reaches to 0.77 after 250 steps iteration.
