# ElasticDL on Google Kubernetes Engine

ElasticDL is a Kubernetes-native machine learning framework.  This document explains how to run an ElasticDL job on Google Kubernetes Engine (GKE).

## Configure Your GKE Environment

To access GKE, we need to install [Google Cloud SDK](https://cloud.google.com/sdk/install), which includes command-line tools like `gcloud`.

- Set the PROJECT_ID environment variable in your shell by retrieving the pre-configured project ID on gcloud by running the command below:

   ```
   export PROJECT_ID="$(gcloud config get-value project -q)"
   ```

- Use the command below to generate the corresponding kubeconfig:

   ```
   gcloud container clusters get-credentials ${PROJECT_ID}
   ```
    and then add the generated config to your local kubeconfig file (`~/.kube/config` by default). 
 
- Make sure you have [`kubectl`](https://kubernetes.io/docs/tasks/tools/install-kubectl/) available locally.

Use the following command to list all the started components.

```bash
kubectl get all --all-namespaces
```

ElasticDL jobs require pod creation and deletion permissions. Make sure you have granted related permissions to the default or other related service accounts.

```bash
kubectl apply -f elasticdl/manifests/examples/elasticdl-rbac.yaml
```

## Build Docker Image

Clone ElasticDL source code:

```bash
git clone https://github.com/wangkuiyi/elasticdl.git
```

Build docker image:

```bash
cd elasticdl
docker build -t gcr.io/${PROJECT_ID}/elasticdl:dev -f elasticdl/docker/Dockerfile .
```

## Upload Docker Image
Configure Docker command-line tool to authenticate to Container Registry:

```
gcloud auth configure-docker
```
and then use the Docker command-line tool to upload the image to your Container Registry:

```
docker push gcr.io/${PROJECT_ID}/elasticdl:dev
```
## Example of Job Submission on GKE
Use the command below to submit your first ElasticDL job on GKE:

```
python -m elasticdl.python.elasticdl.client train \
    --job_name=hello-world \
    --model_zoo=model_zoo \
    --model_def=mnist_subclass.mnist_subclass.CustomModel \
    --training_data_dir=${MNIST_DATA_DIR}/train \
    --evaluation_data_dir=${MNIST_DATA_DIR}/test \
    --num_epochs=1 \
    --minibatch_size=10 \
    --records_per_task=100 \
    --num_workers=2 \
    --checkpoint_steps=2 \
    --grads_to_wait=2 \
    --volume="mount_path=/data,claim_name=fileserver-claim" \
    --log_level=INFO \
    --docker_image_prefix=gcr.io/${PROJECT_ID}
```

where `MNIST_DATA_DIR` is the directory that contains MNIST training and evaluation data in RecordIO format (e.g. /data/mnist_nfs/mnist) and

`--volume` is a string that contains information for the [Kubernetes Volume](https://cloud.google.com/kubernetes-engine/docs/concepts/volumes).


Use the following command to check the job's pods statuses:

```bash
kubectl get pods -l elasticdl-job-name=hello-world
```
You could delete all the pods of the submitted job using the command below:

```
kubectl delete pod -l elasticdl-job-name=hello-world
```

## Example of Job Fault Tolerance
One of the important features of ElasticDL is fault tolerance which ensures job success in extreme cases such as pods get killed due to some reasons.

Same as the first example, submit a job on GKE using the command below:

```
python -m elasticdl.python.elasticdl.client train \
    --job_name=fault-tolerance \
    --model_zoo=model_zoo \
    --model_def=mnist_subclass.mnist_subclass.CustomModel \
    --training_data_dir=${MNIST_DATA_DIR}/train \
    --evaluation_data_dir=${MNIST_DATA_DIR}/test \
    --num_epochs=1 \
    --minibatch_size=10 \
    --records_per_task=100 \
    --num_workers=2 \
    --checkpoint_steps=2 \
    --grads_to_wait=2 \
    --volume="mount_path=/data,claim_name=fileserver-claim" \
    --log_level=INFO \
    --docker_image_prefix=gcr.io/${PROJECT_ID}
```
Check the job's pods statuses and wait until all the pods become `Running`:

```
kubectl get pods -l elasticdl-job-name=fault-tolerance
```
And then delete one of the two worker's pods:

```
kubectl delete pod elasticdl-worker-fault-tolerance-0
```
Keeping track the number of job's pods, you will see the number restores to two pods, and the job will complete successfully.

## Example of Elastic Scheduling
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
python -m elasticdl.python.elasticdl.client train \
    --job_name=low-prio-job \
    --model_zoo=model_zoo \
    --model_def=mnist_subclass.mnist_subclass.CustomModel \
    --training_data_dir=${MNIST_DATA_DIR}/train \
    --evaluation_data_dir=${MNIST_DATA_DIR}/test \
    --master_pod_priority=high-priority \
    --worker_pod_priority=low-priority \
    --num_epochs=1 \
    --minibatch_size=10 \
    --records_per_task=100 \
    --num_workers=2 \
    --checkpoint_steps=2 \
    --master_resource_request="cpu=1,memory=1024Mi" \
    --master_resource_limit="cpu=1,memory=1024Mi" \
    --worker_resource_request="cpu=3,memory=4096Mi" \
    --worker_resource_limit="cpu=3,memory=4096Mi" \
    --grads_to_wait=2 \
    --volume="mount_path=/data,claim_name=fileserver-claim" \
    --log_level=INFO \
    --docker_image_prefix=gcr.io/${PROJECT_ID}
```
Please note that the master pod is configured priority `high-priority` which means the master cannot be preempted even for low priority jobs.

The first job will launch one master pod and two worker pods. Use the following command to check pods statues, and wait until all pods become `Running`.

```bash
kubectl get pods -l elasticdl-job-name=low-prio-job
```

### Submit the second job with `high-priority`
```
python -m elasticdl.python.elasticdl.client train \
    --job_name=high-prio-job \
    --model_zoo=model_zoo \
    --model_def=mnist_subclass.mnist_subclass.CustomModel \
    --training_data_dir=${MNIST_DATA_DIR}/train \
    --evaluation_data_dir=${MNIST_DATA_DIR}/test \
    --master_pod_priority=high-priority \
    --worker_pod_priority=high-priority \
    --num_epochs=1 \
    --minibatch_size=10 \
    --records_per_task=100 \
    --num_workers=1 \
    --checkpoint_steps=2 \
    --master_resource_request="cpu=1,memory=1024Mi" \
    --master_resource_limit="cpu=1,memory=1024Mi" \
    --worker_resource_request="cpu=3,memory=4096Mi" \
    --worker_resource_limit="cpu=3,memory=4096Mi" \
    --grads_to_wait=2 \
    --volume="mount_path=/data,claim_name=fileserver-claim" \
    --log_level=INFO \
    --docker_image_prefix=gcr.io/${PROJECT_ID}
```
Use the following command:

```bash
kubectl get pods -l elasticdl-job-name=high-prio-job
```
You will find the master is Running and a worker is Pending due to insufficient resources.

Because the second job has higher priority than the first one, so soon the first job gets preempted and one of its workers is deleted by Kubernetes, the released resource is re-assigned to the second job.

Because of elastic scheduling, the two ElasticDL jobs continue running.

When the job with high-priority finished, the low-priority job would restore to two pods due to released resources and finish finally.
