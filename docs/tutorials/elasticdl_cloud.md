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

Please refer to [elasticdl_local tutorual](./elasticdl_local.md) to build the `elasticdl:ci` image. The difference is that we have to push the image to Google Cloud repo. We use the following command to get the authentication:

```
gcloud auth configure-docker
```

We launch a training job with 2 PS pods and 4 worker pods. The master pod and PS pods are set with priority, while worker pods are set with low priority.

```
python -m elasticdl.python.elasticdl.client train \
  --image_base=elasticdl:ci \
  --docker_image_repository=gcr.io/${PROJECT_ID}  \
  --model_zoo=model_zoo \
  --model_def=mnist_functional_api.mnist_functional_api.custom_model \
  --training_data=/data/mnist/train \
  --validation_data=/data/mnist/test \
  --num_epochs=5 \
  --master_resource_request="cpu=2,memory=2048Mi" \
  --master_resource_limit="cpu=2,memory=2048Mi" \
  --master_pod_priority=high \
  --worker_resource_request="cpu=2,memory=2048Mi" \
  --worker_resource_limit="cpu=2,memory=2048Mi" \
  --worker_pod_priority=low \
  --ps_resource_request="cpu=2,memory=2048Mi" \
  --ps_resource_limit="cpu=2,memory=2048Mi" \
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


To see the status of each pod:

```
kubectl get pods
```

To see the loss in worker pod:

```
kubectl logs elasticdl-test-mnist-worker-0 | grep "Loss"
```
To see the evaluation metrics in the master pod:

```
kubectl logs elasticdl-test-mnist-master | grep "Evaluation"
```

## Example of Job Fault Tolerance

ElasticDL supports falut tolerance in distributed training. When a worker pod is killed, the master pod will try to relaunch a new worker pod.

At first, all pods are running:

```
elasticdl-test-mnist-master     1/1     Running   0          35s
elasticdl-test-mnist-ps-0       1/1     Running   0          29s
elasticdl-test-mnist-ps-1       1/1     Running   0          28s
elasticdl-test-mnist-worker-0   1/1     Running   0          28s
elasticdl-test-mnist-worker-1   1/1     Running   0          28s
elasticdl-test-mnist-worker-2   1/1     Running   0          28s
elasticdl-test-mnist-worker-3   1/1     Running   0          28s
```

Then, we delete a worker pod:

```
kubectl delete pod elasticdl-test-mnist-worker-0
```

The master pod creates a new worker pod `elasticdl-test-mnist-worker-4` at once.

```
NAME                            READY   STATUS    RESTARTS   AGE
elasticdl-test-mnist-master     1/1     Running   0          51s
elasticdl-test-mnist-ps-0       1/1     Running   0          45s
elasticdl-test-mnist-ps-1       1/1     Running   0          44s
elasticdl-test-mnist-worker-1   1/1     Running   0          44s
elasticdl-test-mnist-worker-2   1/1     Running   0          44s
elasticdl-test-mnist-worker-3   1/1     Running   0          44s
elasticdl-test-mnist-worker-4   1/1     Running   0          6s
```

The training job does not crash, and the new worker pod joins the training smoothly.

## Example of Elastic Scheduling

After we launch the MNIST training job, we launch another nginx service with high priority in the same cluster.

nginx.yaml

```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-nginx
  labels:
    app: nginx
spec:
  replicas: 5
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.7.9
        imagePullPolicy: Always
        ports:
        - containerPort: 80
        resources:
          limits:
            cpu: 2
            memory: 2048Mi
            ephemeral-storage: 1024Mi
          requests:
            cpu: 2
            memory: 2048Mi
            ephemeral-storage: 1024Mi
      priorityClassName: high
      restartPolicy: Always
```

```
kubectl create -f nginx.yaml
```

We will find that some worker pods with low priority are preempted by nginx pods with high priority.

```
NAME                            READY   STATUS        RESTARTS   AGE
elasticdl-test-mnist-master     1/1     Running       0          34s
elasticdl-test-mnist-ps-0       1/1     Running       0          27s
elasticdl-test-mnist-ps-1       1/1     Running       0          27s
elasticdl-test-mnist-worker-0   1/1     Running       0          27s
elasticdl-test-mnist-worker-1   1/1     Terminating   0          27s
elasticdl-test-mnist-worker-2   1/1     Terminating   0          27s
elasticdl-test-mnist-worker-3   1/1     Terminating   0          26s
test-nginx-7585fc5976-5hs7h     1/1     Running       0          2s
test-nginx-7585fc5976-9s4nx     1/1     Running       0          2s
test-nginx-7585fc5976-bf2th     0/1     Pending       0          2s
test-nginx-7585fc5976-ckd94     0/1     Pending       0          2s
test-nginx-7585fc5976-ss8pk     0/1     Pending       0          2s
```

After preemption, the training job still goes on with one worker pod.

```
elasticdl-test-mnist-master     1/1     Running   0          61s
elasticdl-test-mnist-ps-0       1/1     Running   0          54s
elasticdl-test-mnist-ps-1       1/1     Running   0          54s
elasticdl-test-mnist-worker-0   1/1     Running   0          54s
elasticdl-test-mnist-worker-4   0/1     Pending   0          26s
elasticdl-test-mnist-worker-5   0/1     Pending   0          26s
elasticdl-test-mnist-worker-6   0/1     Pending   0          26s
test-nginx-7585fc5976-5hs7h     1/1     Running   0          29s
test-nginx-7585fc5976-9s4nx     1/1     Running   0          29s
test-nginx-7585fc5976-bf2th     1/1     Running   0          29s
test-nginx-7585fc5976-ckd94     1/1     Running   0          29s
test-nginx-7585fc5976-ss8pk     1/1     Running   0          29s
```

Then, we scale the nginx deployment down to 1 replica. Some cluster esources are freed.


```
kubectl scale deployment.v1.apps/test-nginx --replicas=1
```

We find that the training job take over the freed resources, and training with 4 worker pods.

```
NAME                            READY   STATUS    RESTARTS   AGE
elasticdl-test-mnist-master     1/1     Running   0          2m3s
elasticdl-test-mnist-ps-0       1/1     Running   0          116s
elasticdl-test-mnist-ps-1       1/1     Running   0          116s
elasticdl-test-mnist-worker-0   1/1     Running   0          116s
elasticdl-test-mnist-worker-4   1/1     Running   0          88s
elasticdl-test-mnist-worker-5   1/1     Running   0          88s
elasticdl-test-mnist-worker-6   1/1     Running   0          88s
test-nginx-7585fc5976-5hs7h     1/1     Running   0          91s

```
