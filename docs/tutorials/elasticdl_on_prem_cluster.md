# ElasticDL on On-prem Cluster

## Environment Preparation

We should install ElasticDL first. Please refer to the installation part in
[elastic_local](elasticdl_local.md) doc.

Then, build needed images.

```bash
export TRAVIS_BUILD_DIR=$PWD
bash scripts/travis/build_images.sh
```

## Submit Job to Cluster

The submit command is similar to local mode. The local scripts will be built
into a docker image, and pushed to `$DOCKER_HUB_REPO` remote docker hub.

Following is an exmaple:

```bash
export DOCKER_HUB_REPO=reg.docker.com/user/
```

```bash
elasticdl train \
 --image_base=elasticdl:ci \
 --docker_image_prefix=$DOCKER_HUB_REPO \
 --model_zoo=./model_zoo \
 --model_def=mnist_functional_api.mnist_functional_api.custom_model \
 --training_data=/data/mnist/train \
 --validation_data=/data/mnist/test \
 --num_epochs=2 \
 --master_resource_request="cpu=1,memory=2048Mi,ephemeral-storage=5000Mi" \
 --worker_resource_request="cpu=1,memory=2048Mi,ephemeral-storage=5000Mi" \
 --minibatch_size=64 \
 --num_minibatches_per_task=2 \
 --num_workers=2 \
 --checkpoint_steps=10 \
 --grads_to_wait=2 \
 --job_name=test-mnist \
 --log_level=INFO \
 --image_pull_policy=Always \
 --namespace=kubemaker
```

Then the job will be launched on the cluster.

By the way, we can also use a pre-built image to submit the ElasticDL job.

```bash
elasticdl train \
 --image_base=reg.docker.com/user/elasticdl:mnist \
 --model_zoo=/model_zoo \
 --model_def=mnist_functional_api.mnist_functional_api.custom_model \
 --training_data=/data/mnist/train \
 --validation_data=/data/mnist/test \
 --num_epochs=2 \
 --master_resource_request="cpu=1,memory=2048Mi,ephemeral-storage=5000Mi" \
 --worker_resource_request="cpu=1,memory=2048Mi,ephemeral-storage=5000Mi" \
 --minibatch_size=64 \
 --num_minibatches_per_task=2 \
 --num_workers=2 \
 --checkpoint_steps=10 \
 --grads_to_wait=2 \
 --job_name=test-mnist \
 --log_level=INFO \
 --image_pull_policy=Always \
 --namespace=kubemaker
```

## Add Cluster-specific Information

If the on-prem cluster requires additional specifications to pods or services,
such as labels, tolerations, etc, we can add an additional argument
`--cluster_spec spec.py` in the command line above. We define a class instance
`cluster` in `spec_py` file. There are two required class functions `with_pod`
and `with_service` for adding additional specifications to pods or services.

Below is an example of `spec.py`.

```python
from kubernetes import client

class MyCluster:
    def __init__(self):
        self._pool = "elasticdl"
        self._app_name = "elasticdl"

    # Add pod specifications
    def with_pod(self, pod):
        # Add a label
        pod.metadata.labels["my_app"] = self._app_name

        # Add tolerations
        tolerations = [
            client.V1Toleration(
                effect="NoSchedule",
                key="mycluster.com/app-pool",
                operator="Equal",
                value=self._pool ,
            ),
        ]
        pod.spec.tolerations = tolerations
        return pod

    # Add service specifications
    def with_service(self, service):
        # Use ClusterIP
        service.spec.type = "ClusterIP"
        service.spec.cluster_ip = "None"
        return service

cluster = MyCluster()
```
