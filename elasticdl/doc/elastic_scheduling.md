# Elastic Scheduling For ElasticDL Training Job
TODO: A summary of the background of Elastic Scheduling.

## Configure Local GKE Environment
### Install required softwares
* Install **python v2.7** which is required by gcloud CLI ([miniconda](https://docs.conda.io/en/latest/miniconda.html) is recommended for managing multi-version python environment).
* Install [gcloud CLI](https://cloud.google.com/sdk/docs/quickstart-macos).
* Merge the google cloud kubernetes cluster config into your mac’s `/Users/${user}/.kube/config` to make the local kubectl could access GKE. 

### Check the GKE cluster
* The GKE cluster portal: [cluster portal](https://console.cloud.google.com/home/dashboard?project=${project_name})
* The docker image list: [docker images](https://console.cloud.google.com/gcr/images/elasticdl/GLOBAL?project=${project_name})
* All the started components can be viewed through the command:

```
kubectl get all --all-namespaces
```

### Bind service account
For the master have the permission to call kubernetes core API, the service account must be binded with role cluster-admin. create yaml file rbac.yaml with the content below:

```
apiVersion: rbac.authorization.k8s.io/v1beta1
kind: ClusterRoleBinding
metadata:
  name: rbac
subjects:
  - kind: ServiceAccount
    name: default
    namespace: default
roleRef:
  kind: ClusterRole
  name: cluster-admin
  apiGroup: rbac.authorization.k8s.io
```
and then execute command:

```
kubectl apply -f rbac.yaml
```
### Identity verification for container registry
Execute the command below:

```
gcloud auth configure-docker
```

## Prepare Docker Image
### Download elasticdl source code
```bash
git clone https://github.com/wangkuiyi/elasticdl.git
```
### Build docker images
```bash
cd elasticdl
docker build -t gcr.io/${project_name}/elasticdl:dev -f elasticdl/docker/Dockerfile .
```
## Test Case For Elastic Scheduling
Assume we have a GKE cluster of three vm instances and resource specification of  each vm is:

```
CPU: 4 vCPU
Memory: 15GB

```
### Setup priority classes
Kubernetes provide job with priority through PriorityClass. To test the ability of elastic scheduling, execute the commands below:

```
kubectl apply -f high-prio.yaml
```
and the content of high-prio.yaml is: 

```
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority
value: 1000000
globalDefault: false
```
```
kubectl apply -f low-prio.yaml
```
and the content of low-prio.yaml is: 

```
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: low-priority
value: 1000
globalDefault: false
```

More about PriorityClass, see [Pod Priority and Preemption](https://kubernetes.io/docs/concepts/configuration/pod-priority-preemption/).

### Submit the first elasticdl job with pod priority `low-priority`
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
Make sure the pods of a single master and two worker's statuses are Running using the command:

```
kubectl get pod
```

### Submit the second elasticdl job with pod priority `high-priority`
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
Using the command `kubectl get pod` you will see the status of master is Running and the status of the single worker is Pending because of insufficient resources, because the priority of the second job is higher than the first job, so soon the first job is preempted and one of the two workers of the first job is deleted by kubernetes, the released resource is assigned to the second job. 
because the ability of elastic scheduling, the two elasticdl jobs will become finished at last.