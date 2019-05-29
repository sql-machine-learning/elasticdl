# ElasticDL on Google Kubernetes Engine
This document illustrates how to run ElasticDL with elastic scheduling in Google Kubernetes Engine (GKE) environment. Before we start, make sure you have a GKE account and a running cluster there.

## Configure You GKE Environment

To access GKE, we need to install the following tools.

* Install [gcloud CLI](https://cloud.google.com/sdk/docs/quickstart-macos). Note that gcloud CLI requires **python v2.7**. [miniconda](https://docs.conda.io/en/latest/miniconda.html) is recommended for managing multi-version python environment.
* Use the command below to generate corresponding kubeconfig:

   ```
   gcloud container clusters get-credentials ${cluster_name}
   ```
    and then add the generated config to your local kubeconfig file (`~/.kube/config` by default). 
 
* Make sure you have [`kubectl`](https://kubernetes.io/docs/tasks/tools/install-kubectl/) available locally.

Use the following command to list all the started components.
```bash
kubectl get all --all-namespaces
```

ElasticDL jobs require pod creation and deletion permissions. Make sure you have grantted related permissions to the default or other related service accounts.
```bash
kubectl apply -f elasticdl/elasticdl/manifests/examples/manifests/examples/elasticdl-rbac.yaml
```

## Buil Docker Image

Clone ElasticDL source code:
```bash
git clone https://github.com/wangkuiyi/elasticdl.git
```

Build docker image:

```bash
cd elasticdl
docker build -t gcr.io/${project_name}/elasticdl:dev -f elasticdl/docker/Dockerfile .
```

## Upload Docker Image
Configure Docker command-line tool to authenticate to Container Registry:

```
gcloud auth configure-docker
```
and then use the Docker command-line tool to upload the image to your Container Registry:

```
docker push gcr.io/${project_name}/elasticdl:dev
```


## Test Case For Elastic Scheduling
Assume we have a GKE cluster with three instances, and each instance is configured with 4 CPU cores and 15 GB memory.

### Setup priority classes

Kubernetes provides priority for jobs using PriorityClass. To test the ability of elastic scheduling, you need to create two customized PriorityClass. Save the following two yaml files as high-prio.yaml and low-prio.yaml respectively.

```
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority
value: 1000000
globalDefault: false
```
```
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: low-priority
value: 1000
globalDefault: false
```
And then execute the commands below to create PriorityClass in GKE cluster:

```
kubectl apply -f high-prio.yaml
kubectl apply -f low-prio.yaml
```
For more about PriorityClass, please check out [Pod Priority and Preemption](https://kubernetes.io/docs/concepts/configuration/pod-priority-preemption/).

### Submit the first job with `low-priority`
```
python elasticdl/python/elasticdl/client/client.py \
    --job_name=low-prio-job \
    --model_file=elasticdl/python/examples/mnist_functional_api.py \
    --train_data_dir=/data/mnist/train \
    --master_pod_priority=high-priority \
    --worker_pod_priority=low-priority \
    --num_epoch=1 \
    --minibatch_size=10 \
    --record_per_task=100 \
    --num_worker=2 \
    --master_cpu_request=1000m \
    --master_cpu_limit=1000m \
    --master_memory_request=1024Mi \
    --master_memory_limit=1024Mi \
    --worker_cpu_request=3000m \
    --worker_cpu_limit=3000m \
    --worker_memory_request=4096Mi \
    --worker_memory_limit=4096Mi \
    --grads_to_wait=2 \
    --codec_type=bytes \
    --repository=gcr.io \
    --image_base=gcr.io/elasticdl/elasticdl:dev
```

The first job will launch one master pod and two worker pods. Use the following command to check pods statues, and wait until all pods become `Running`.

```bash
kubectl get pods -l elasticdl_job_name=low-prio-job
```

### Submit the second job with `high-priority`
```
python elasticdl/python/elasticdl/client/client.py \
    --job_name=high-prio-job \
    --model_file=elasticdl/python/examples/mnist_functional_api.py \
    --train_data_dir=/data/mnist/train \
    --master_pod_priority=high-priority \
    --worker_pod_priority=high-priority \
    --num_epoch=1 \
    --minibatch_size=10 \
    --record_per_task=100 \
    --num_worker=1 \
    --master_cpu_request=1000m \
    --master_cpu_limit=1000m \
    --master_memory_request=1024Mi \
    --master_memory_limit=1024Mi \
    --worker_cpu_request=3000m \
    --worker_cpu_limit=3000m \
    --worker_memory_request=4096Mi \
    --worker_memory_limit=4096Mi \
    --grads_to_wait=2 \
    --codec_type=bytes \
    --repository=gcr.io \
    --image_base=gcr.io/elasticdl/elasticdl:dev
```
Use the following command:

```bash
kubectl get pods -l elasticdl_job_name=high-prio-job
```
You will find the master is Running and a worker is Pending due to insufficient resources.

Because the second job has higher priority than the first one, so soon the first job gets preempted and one of its workers is deleted by Kubernetes, the released resource is re-assigned to the second job.
 
Because of elastic scheduling, the two elasticdl jobs will finish finally.


