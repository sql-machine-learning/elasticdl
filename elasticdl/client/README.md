# ElasticDL Client: Submit ElasticDL Job to Kubernetes 

Currently for Mac docker-for-desktop only.

## Check Environment

Make sure the Kubernetes docker-for-desktop (not minikube) is installed on your Mac.

## Download ElasticDL Source Code
```bash
git clone https://github.com/wangkuiyi/elasticdl.git
cd elasticdl
```

## Build ElasticDL Development Docker Image
```bash
docker build -t elasticdl:dev -f dockerfile/elasticdl.dev .
```
The Kubernetes example use `elasticdl:dev` Docker image as the base master/worker image.


## Write a Keras Model

**(TODO: Describe programming API)**

There are several Keras examples provided in `edl_k8s_examples` directory.

## Submit EDL job

To submit a model, e.g. `edl_k8s_examples/mnist_model.py` to ElasticDL system:

```bash
python elasticdl/client/client.py \
    --model_file=edl_k8s_examples/mnist_model.py \
    --train_data_dir=/data/mnist/train \
    --num_epoch=1 \
    --minibatch_size=10 \
    --record_per_task=100 \
    --num_worker=1 \
    --grads_to_wait=2
```

## Check the pod status

```bash
kubectl get pods
kubectl logs ${pod_name}
```
