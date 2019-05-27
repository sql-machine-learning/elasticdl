# ElasticDL: A Kubernetes-native Deep Learning Framework

## The Development Docker Image

To build the development Docker image, in repo's root directory, run the following command:

```bash
docker build \
    -t elasticdl:dev \
    -f elasticdl/docker/Dockerfile .
```

When having difficulties downloading from the main PYPI site, You could pass an extra PYPI index url to `docker build`, such as:

```bash
docker build \
    --build-arg EXTRA_PYPI_INDEX=https://mirrors.aliyun.com/pypi/simple \
    -t elasticdl:dev \
    -f elasticdl/docker/Dockerfile .
```


To develop in the Docker container, run the following command to mount your cloned `elasticdl` git repo directory (e.g. `EDL_REPO` below) to `/elasticdl` directory in the container and start container:

```bash
EDL_REPO=<your_elasticdl_git_repo>
docker run --rm -u $(id -u):$(id -g) -it \
    -v $EDL_REPO:/v \
    -w /v \
    elasticdl:dev
```

## Test and Debug

### Unittests

In dev Docker container's `elasticdl` repo's root directory, do the following:

```bash
make && python -m unittest discover elasticdl/python '*_test.py'
```

Could also start Docker container and run unittests in a single command:

```bash
docker run --rm -u $(id -u):$(id -g) -it \
    -v $EDL_REPO:/v \
    -w /v \
    elasticdl:dev \
    bash -c "make && python -m unittest discover elasticdl/python '*_test.py'"
```
### Test in Docker

In a terminal, start master to distribute mnist training tasks.

```
docker run --net=host --rm -it elasticdl:dev \
    bash -c "cd elasticdl/python &&
      python -m elasticdl.master.main \
          --model_file=examples/mnist_functional_api.py \
          --job_name=test \
          --train_data_dir=/data/mnist/train \
          --record_per_task=100 \
          --num_epoch=2 \
          --codec_type=tf_example \
          --grads_to_wait=2 \
          --minibatch_size=10"
```

In another terminal, start a worker

```
docker run --net=host --rm -it elasticdl:dev \
    bash -c "cd elasticdl/python &&
      python -m elasticdl.worker.main \
          --worker_id=1 \
          --model_file=examples/mnist_functional_api.py \
          --codec_type=tf_example \
          --master_addr=localhost:50001"
```

This will train MNIST data with a model defined in [python/examples/mnist_functional_api.py](python/examples/mnist_functional_api.py) for 2 epoches.

### Test with Kubernetes

We can also test ElasticDL job in a Kubernetes environment using the previous built [image](#the-development-docker-image).

For Minikube, run the following command to launch the job.
```bash
kubectl apply -f manifests/examples/elasticdl-demo-minikube.yaml
```
Note that, to let Minikube use local image, you need to run `eval $(minikube docker-env)` first, and then build the image following [instructions](#the-development-docker-image).

For other Kubernetes clusters, first make sure the built image has been pushed to some registries, and then run the following command to launch the job. 
```bash
kubectl apply -f manifests/examples/elasticdl-demo-k8s.yaml
```

If you find permission error in the main pod log, e.g., `"pods is forbidden: User \"system:serviceaccount:default:default\" cannot create resource \"pods\""`, you need to grant pod-related permissions for the default user.
```bash
kubectl apply -f manifests/examples/elasticdl-rbac.yaml
```

### Manual Debug

Sometimes it is easier to debug with a real master server. To start master server in container, run the following in `elasticdl` directory:

```bash
make && python -m master.main
```

### Run ElasticDL On GKE
https://docs.google.com/document/d/1cbzYjHTd7SUAPUGbT82AcmVu_4iWWjwykrryIDH-fOU/edit#heading=h.mbjsiz6n6jlo

### ElasticDL Priority-Based Elastic Scheduling
https://docs.google.com/document/d/1H4rhRc5Ll0uxTkiVf2fV_Q0xFf0HlKas1IAqPtWGHcQ/edit#heading=h.vb8p0lepu9vn (Note: The doc is currently in Chinese.)
