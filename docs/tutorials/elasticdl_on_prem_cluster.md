# ElasticDL on On-prem Cluster

## Environment preparation

You should install ElasticDL first. Please refer to the installation part in [elastic_local](elasticdl_local.md) doc.

Then, build needed images.

```bash
bash elasticdl/docker/build_all.sh
```

## Submit job to cluster

The submit command is similar to local mode. The local scripts will be built into a docker image, and pushed to `$DOCKER_HUB_REPO` remote docker hub.

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

By the way, we can also use the pre-built image to submit the ElasticDL job.

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
