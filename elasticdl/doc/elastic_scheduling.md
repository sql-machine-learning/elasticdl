# Elastic Scheduling For ElasticDL Training Job
This doc illustrates how to run ElasticDL with elastic scheduling in Google Kubernetes Engine (GKE) environment. Before we start, make sure you have a GKE account and a running cluster there.

## Configure You GKE Environment

To access GKE from your local environment, we need to install some tools.

* Install [gcloud CLI](https://cloud.google.com/sdk/docs/quickstart-macos). Note that gcloud CLI requires **python v2.7**. [miniconda](https://docs.conda.io/en/latest/miniconda.html) is recommended for managing multi-version python environment.
* Use the command below to generate corresponding kubeconfig:

   ```
gcloud container clusters get-credentials ${cluster_name}
```
 and then add the generated config to your local kubeconfig file (`~/.kube/config` by default). 
 
* Make sure you have [`kubectl`](https://kubernetes.io/docs/tasks/tools/install-kubectl/) available locally.

Now you can access the GKE cluster from your environment. Use the following command to list all started components in the cluster.
```bash
kubectl get all --all-namespaces
```

ElastciDL job requires permissions to create and delete pods. Make sure you grant related permissions to default or other related service account.
```bash
kubectl apply -f ../manifests/examples/elasticdl-rbac.yaml
```

## Build The Docker Image

Download ElasticDL source code:
```bash
git clone https://github.com/wangkuiyi/elasticdl.git
```

Build docker image:
```bash
cd elasticdl
docker build -t gcr.io/${project_name}/elasticdl:dev -f elasticdl/docker/Dockerfile .
```

## Upload The Docker Image
First, configure Docker command-line tool to authenticate to Container Registry:

```
gcloud auth configure-docker
```
and then use the Docker command-line tool to upload the image to your Container Registry:

```
docker push gcr.io/${project_name}/elasticdl:dev
```


## Test Case For Elastic Scheduling
Assume we have a GKE cluster with three instances. Each instance is configured with 4 CPU cores and 15 GB memory.

### Setup priority classes

Kubernetes provide job with priority through PriorityClass. To test the ability of elastic scheduling, you should create two customized PriorityClass. The first step is to save the following two yaml file as high-prio.yaml and low-prio.yaml respectively.

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

### Submit the first ElasticDL job with `low-priority`
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

### Submit the second ElasticDL job with `high-priority`
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
Using the command `kubectl get pod` you will see the status of master is Running and the status of the single worker is Pending because of insufficient resources, because the priority of the second job is higher than the first job, so soon the first job is preempted and one of the two workers of the first job is deleted by Kubernetes, the released resource is assigned to the second job. 
Because of elastic scheduling, the two elasticdl jobs will finish finally.

