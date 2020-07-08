# ElasticDL on On-prem Cluster

## Environment Preparation

In order to find and access the on-premise cluster, ElasticDL needs a
[kubeconfig file](https://kubernetes.io/docs/tasks/access-application-cluster/configure-access-multiple-clusters)
, which is located at `~/.kube/config` by default.

We also need to install ElasticDL client.

```bash
pip install elasticdl-client
```

## Submit Job to Cluster

The job submission steps are similar to public cloud mode. Please
refer to [Submit Job](elasticdl_cloud.md#submit-job-to-the-kubernetes-cluster)
section in [ElasticDL on Public Cloud tutorial](elasticdl_cloud.md)
for detail. The difference is that we are not restricted to google cloud
repo. So we can push the image to any remote docker hub that the on-premise
cluster can access.

```bash
export DOCKER_HUB_REPO=reg.docker.com/user/

cd ${CODE_PATH}/elasticdl/model_zoo

elasticdl zoo init

elasticdl zoo build --image=${DOCKER_HUB_REPO}/elasticdl:mnist .

elasticdl zoo push ${DOCKER_HUB_REPO}/elasticdl:mnist
```

We launch a training job with 2 PS pods and 4 worker pods.

```bash
elasticdl train \
  --image_name=${DOCKER_HUB_REPO}/elasticdl:mnist \
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

## Add Cluster-Specific Information

If the on-premise cluster is a tailored version of Kubernetes which
requires additional labels, or we need to add tolerations or node affinity
to the job's pods, we can use an additional argument
`--cluster_spec spec.py` in the command line above. We define a class instance
`cluster` in `spec.py` file. There are two required class functions `with_pod`
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
