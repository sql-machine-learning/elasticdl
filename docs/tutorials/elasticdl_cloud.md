# ElasticDL on Public Cloud

ElasticDL is a Kubernetes-native machine learning framework.  This document explains how to run an ElasticDL job on a public cloud, namely, Google Kubernetes Engine (GKE).

## Configure GKE Environment

### Create a Project and a Kubernetes Cluster

First, we create a new project for elasticdl in [web console](https://console.cloud.google.com/) and a new Kubernetes cluster under this project.

We will use the project id and cluster name in next steps.

### Access the Kubernetes Cluster

To access GKE, we need to install [Google Cloud SDK](https://cloud.google.com/sdk/install), which includes command-line tools like `gcloud`.


1. Set the PROJECT_ID environment variable in shell

```
export PROJECT_ID=${your_project_id}
gcloud config set project ${PROJECT_ID}
```

1. List clusters info with gcloud, and double check it with web console

```
gcloud container clusters list
```

Following is an our testing cluster

```
NAME         LOCATION       MASTER_VERSION  MASTER_IP       MACHINE_TYPE   NODE_VERSION    NUM_NODES  STATUS
edl-cluster  us-central1-c  1.14.10-gke.36  x.x.x.x         n1-standard-8  1.14.10-gke.36  3          RUNNING
```

1. Use the command below to generate the corresponding kubeconfig

```
gcloud container clusters get-credentials edl-cluster --zone us-central1-c
```
 
1. Make sure you have [`kubectl`](https://kubernetes.io/docs/tasks/tools/install-kubectl/) available locally.

Use the following command to list all the started components.

```
kubectl get all --all-namespaces
```


### Config the Kubernetes Cluster

ElasticDL jobs require pod creation and deletion permissions. Make sure you have granted related permissions to the default or other related service accounts.

```bash
kubectl apply -f elasticdl/manifests/examples/elasticdl-rbac.yaml
```

ElasticDL supports elastic scheduling, and works well the priority-based scheduling of Kubernetes. We create two customized PriorityClass in this testing cluster, high and low.


high.yaml

```
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high
value: 1000000
globalDefault: false
```

low.yaml

```
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: low
value: 1000
globalDefault: false
```

### Mount a Volume for the Kubernetes Cluster

First, we create a [Cloud Filestore](https://cloud.google.com/filestore) instance in web console.

Then we follow the [doc](https://cloud.google.com/filestore/docs/accessing-fileshares) to access fileshares from the Kubernetes cluster.


## Submit Job to the Kubernetes Cluster

We submit a mnist job.


### Prepare Dataset

- We generate MNIST training and evaluation data in recordio format

```
python elasticdl/python/data/recordio_gen/image_label.py --dataset mnist --records_per_shard 4096 .
```

- We launch a pod which mounts the volume, and use `kubectl cp` command to copy data from local to the volume

```
kubectl create -f my-pod.yaml

kubectl cp mnist my-pod:/data
```

my-pod.yaml

```
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
  - name: test-pod
    image: nginx:1.7.9
    volumeMounts:
    - mountPath: /data
      name: mypvc
  volumes:
  - name: mypvc
    persistentVolumeClaim:
      claimName: fileserver-claim
      readOnly: false
```



### Submit Job

Please refer to elasticdl_local.md to build the elasticdl:ci image. The difference is that we have to push the image to Google Cloud repo. We authenticate to Container Registry:

```
gcloud auth configure-docker
```

Submit the training job:


```
python -m elasticdl.python.elasticdl.client train \
  --image_base=elasticdl:ci \
  --docker_image_repository=gcr.io/${PROJECT_ID}  \
  --model_zoo=model_zoo \
  --model_def=mnist_functional_api.mnist_functional_api.custom_model \
  --training_data=/data/mnist/train \
  --validation_data=/data/mnist/test \
  --num_epochs=2 \
  --master_resource_request="cpu=1,memory=2048Mi" \
  --master_resource_limit="cpu=1,memory=2048Mi" \
  --master_pod_priority=high \
  --worker_resource_request="cpu=1,memory=2048Mi" \
  --worker_resource_limit="cpu=1,memory=2048Mi" \
  --worker_pod_priority=low \
  --ps_resource_request="cpu=1,memory=2048Mi" \
  --ps_resource_limit="cpu=1,memory=2048Mi" \
  --ps_pod_priority=high \
  --minibatch_size=64 \
  --num_minibatches_per_task=64 \
  --num_ps_pods=2 \
  --num_workers=4 \
  --evaluation_steps=200 \
  --grads_to_wait=1 \
  --job_name=test-mnist \
  --log_level=INFO \
  --image_pull_policy=Always \
  --volume="mount_path=/data,claim_name=fileserver-claim" \
  --distribution_strategy=ParameterServerStrategy
```

After submitting the job to the cluster, we can run the following command to check the status of each pod:


```
kubectl get pods
```

We will see the status of each pod:

```
NAME                            READY   STATUS    RESTARTS   AGE
elasticdl-test-mnist-master     1/1     Running   0          41s
elasticdl-test-mnist-ps-0       1/1     Running   0          33s
elasticdl-test-mnist-ps-1       1/1     Running   0          32s
elasticdl-test-mnist-worker-0   1/1     Running   0          32s
elasticdl-test-mnist-worker-1   1/1     Running   0          32s
elasticdl-test-mnist-worker-2   1/1     Running   0          32s
elasticdl-test-mnist-worker-3   1/1     Running   0          32s
```

To see the loss in worker pod:

```
kubectl logs elasticdl-test-mnist-worker-0 | grep "Loss"
```
We will see following logs:

```
[2020-06-08 02:54:01,489] [INFO] [worker.py:887:_process_minibatch] Loss = 2.906989812850952, steps = 0
[2020-06-08 02:54:35,487] [INFO] [worker.py:887:_process_minibatch] Loss = 0.5924279689788818, steps = 100
[2020-06-08 02:55:06,928] [INFO] [worker.py:887:_process_minibatch] Loss = 0.46202218532562256, steps = 201
[2020-06-08 02:55:40,727] [INFO] [worker.py:887:_process_minibatch] Loss = 0.26237753033638, steps = 300
```

To see the evaluation metrics in the master pod:

```
kubectl logs elasticdl-test-mnist-master | grep "Evaluation"
```

We will see following logs:

```
[2020-06-08 02:53:46,884] [INFO] [master.py:195:prepare] Evaluation service started
[2020-06-08 02:55:13,525] [INFO] [evaluation_service.py:214:complete_task] Evaluation metrics[v=200]: {'accuracy': 0.8066}
[2020-06-08 02:56:20,930] [INFO] [evaluation_service.py:214:complete_task] Evaluation metrics[v=400]: {'accuracy': 0.8973}
```



## Example of Job Fault Tolerance

TODO

## Example of Elastic Scheduling

TODO
