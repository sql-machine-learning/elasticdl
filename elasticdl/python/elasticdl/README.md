# ElasticDL Client: Submit ElasticDL Job to Kubernetes 

Currently for Mac docker-for-desktop only.

## Check Environment

Make sure the Kubernetes docker-for-desktop (not minikube) is installed on your Mac.


## Prepare Model Definition

A model definition directory is needed to be created, the files in the directory are as follows:

* (mandatory) A Python source file which defines the keras model and use the directory base name as the filename.
* (mandatory) The file `__init__.py` is necessary.
* (optional) Source files of other Python modules.
* (optional) A requirements.txt file that lists dependencies required by the above source files.


There are several Keras examples provided in `elasticdl/python/examples` directory.

## Submit ElasticDL Job In Development Mode

### Download ElasticDL Source Code
```bash
git clone https://github.com/wangkuiyi/elasticdl.git
cd elasticdl
```

Use ElasticDL client to launch ElasticDL system on a Kubernetes cluster and submit a model, e.g. `elasticdl/python/examples/mnist_subclass/mnist_subclass.py` to it.

### Submit to local Kubernetes on Your Machine

For demonstration purposes, we use the data stored on `elasticdl:ci` Docker image.

First we build all development Docker images, which include `elasticdl:ci` image:

```bash
elasticdl/docker/build_all.sh
```

Submit training job:

```bash
python -m elasticdl.python.elasticdl.client train \
    --model_def=elasticdl/python/examples/mnist_subclass \
    --image_base=elasticdl:ci \
    --training_data_dir=/data/mnist/train \
    --evaluation_data_dir=/data/mnist/test \
    --num_epochs=1 \
    --master_resource_request="cpu=1,memory=512Mi" \
    --master_resource_limit="cpu=1,memory=512Mi" \
    --worker_resource_request="cpu=1,memory=1024Mi" \
    --worker_resource_limit="cpu=1,memory=1024Mi" \
    --minibatch_size=10 \
    --records_per_task=100 \
    --num_workers=1 \
    --checkpoint_steps=2 \
    --grads_to_wait=2 \
    --job_name=test \
    --image_pull_policy=Never \
    --log_level=INFO
```

### Submit to a GKE cluster

Please checkout [this tutorial](../../doc/elastic_scheduling.md) for instructions on submitting jobs to a GKE cluster.

### Submit to an on-premise Kubernetes cluster

On-premise Kubernetes cluster may add some additional configurations for pods to be launched,
ElasticDL provides an easy way for users to specify their pods requirements.

```bash
python -m elasticdl.python.elasticdl.client train \
    --job_name=test \
    --image_name=gcr.io/elasticdl/mnist:dev \
    --model_def=elasticdl/python/examples/mnist_subclass \
    --cluster_spec=<path_to_cluster_specification_file> \
    --training_data_dir=/data/mnist_nfs/mnist/train \
    --evaluation_data_dir=/data/mnist_nfs/mnist/test \
    --num_epochs=1 \
    --minibatch_size=10 \
    --records_per_task=100 \
    --num_workers=1 \
    --checkpoint_steps=2 \
    --master_pod_priority=high-priority \
    --worker_pod_priority=high-priority \
    --master_resource_request="cpu=1,memory=2048Mi" \
    --master_resource_limit="cpu=1,memory=2048Mi" \
    --worker_resource_request="cpu=2,memory=4096Mi" \
    --worker_resource_limit="cpu=2,memory=4096Mi" \
    --grads_to_wait=2 \
    --volume="mount_path=/data,claim_name=fileserver-claim" \
    --image_pull_policy=Always \
    --log_level=INFO \
    --docker_image_prefix=gcr.io/elasticdl
```

The difference is that we add a new argument `cluster_spec` which points to a cluster specification file.
The cluster specification module includes a `cluster` component, and ElasticDL will invoke function
`cluster.with_cluster(pod)` to add extra specifications to the 
[pod](https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/V1Pod.md). Here is an example
which return the `pod` directly with no additional specifications. Users can implement more customized configurations
inside the `with_cluster` function.

```python
class KubernetesCluster:
    def with_cluster(self, pod):
        # By default, don't need to add specific config for pod
        return pod

# TODO: need to change this after we make same change to model definition
cluster = KubernetesCluster()
```

## Submit ElasticDL Job In Command Line Mode

### Download ElasticDL Source Code And Build Wheel Package
```bash
git clone https://github.com/wangkuiyi/elasticdl.git
cd elasticdl
```

### Build And Install Wheel Package From Source Code
```bash
python3 setup.py bdist_wheel
pip install dist/elasticdl-0.0.1-py3-none-any.whl
```

### Submit Jobs

Same as in the development mode, just replace `python -m elasticdl.python.elasticdl.client` part with `elasticdl`.

## Check the pod status

```bash
kubectl get pods
kubectl logs $pod_name
```
