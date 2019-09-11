# ElasticDL on cluster

## Environment preparation

You should install ElasticDL first. Please refer to the installation part in [elastic_local](./elasticdl_local.md) doc.

Then, please build needed images.

```bash
bash elasticdl/docker/build_all.sh
```

## Submit job to cluster

The submit command is similar to local mode. Following is an exmaple:


```bash
python -m elasticdl.python.elasticdl.client train \
 --image_base=elasticdl:ci \
 --cluster_spec=$CLUSTER_SPEC \
 --model_zoo=./model_zoo \
 --docker_image_prefix=$DOCKER_HUB_REPO \
 --model_def=mnist_functional_api.mnist_functional_api.custom_model \
 --training_data_dir=/data/mnist/train \
 --evaluation_data_dir=/data/mnist/test \
 --num_epochs=2 \
 --master_resource_request="cpu=1,memory=2048Mi,ephemeral-storage=5000Mi" \
 --worker_resource_request="cpu=1,memory=2048Mi,ephemeral-storage=5000Mi" \
 --minibatch_size=64 \
 --records_per_task=100 \
 --num_workers=2 \
 --checkpoint_steps=10 \
 --grads_to_wait=2 \
 --job_name=test-mnist \
 --log_level=INFO \
 --image_pull_policy=Always \
 --namespace=kubemaker
```

It will build a image locally and push to the remote docker hub. And then the job will be launched on the cluster.