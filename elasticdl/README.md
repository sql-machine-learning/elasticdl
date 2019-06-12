# ElasticDL: A Kubernetes-native Deep Learning Framework

## The Development Docker Image

Development Docker image contains ElasticDL system code and processed demo data in RecordIO format. We first build the demo data image. This only need to be built once.

```bash
docker build \
    -t elasticdl:data \
    -f elasticdl/docker/Dockerfile.data .
```

When building the development Docker image, the data will be copied from data image to development image. Development image needs to be rebuilt when there are code changes. In repo's root directory, run the following command:

```bash
docker build \
    -t elasticdl:dev \
    -f elasticdl/docker/Dockerfile .
```

To build the Docker image with GPU support, run the following command:

```bash
docker build \
    -t elasticdl:dev-gpu \
    -f elasticdl/docker/Dockerfile \
    --build-arg BASE_IMAGE=tensorflow/tensorflow:1.13.1-gpu-py3 .
```

If you are running the examples in the repo, the datasets need to be generated when building the image, which
could be achieved by adding `GEN_DATA=yes` to the build arg like the following:

```bash
docker build \
    --build-arg GEN_DATA=yes \
    -t elasticdl:dev \
    -f elasticdl/docker/Dockerfile .
```

When having difficulties downloading from the main PYPI site, you could pass an extra PYPI index url to `docker build`, such as:

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
make && K8S_TESTS=False pytest elasticdl/python/tests
```

Could also start Docker container and run unittests in a single command:

```bash
docker run --rm -u $(id -u):$(id -g) -it \
    -v $EDL_REPO:/v \
    -w /v \
    elasticdl:dev \
    bash -c "make && K8S_TESTS=False pytest elasticdl/python/tests"
```

Note that, some unittests may require a running Kubernetes cluster available. To include those unittests, use:

```bash
make && pytest elasticdl/python/tests
```

### Test in Docker

In a terminal, start master to distribute mnist training tasks.

```
docker run --net=host --rm -it elasticdl:dev \
    bash -c "python -m elasticdl.python.elasticdl.master.main \
          --model_file=elasticdl/python/examples/mnist_functional_api.py \
          --job_name=test \
          --training_data_dir=/data/mnist/train \
          --evaluation_data_dir=/data/mnist/test \
          --records_per_task=100 \
          --num_epochs=2 \
          --codec_type=tf_example \
          --grads_to_wait=2 \
          --minibatch_size=10 \
          --log_level=INFO"
```

In another terminal, start a worker

```
docker run --net=host --rm -it elasticdl:dev \
    bash -c "python -m elasticdl.python.elasticdl.worker.main \
          --worker_id=1 \
          --model_file=elasticdl/python/examples/mnist_functional_api.py \
          --codec_type=tf_example \
          --master_addr=localhost:50001 \
          --log_level=INFO"
```

This will train MNIST data with a model defined in [python/examples/mnist_functional_api.py](python/examples/mnist_functional_api.py) for 2 epoches.

### Test with Kubernetes

We can also test ElasticDL job in a Kubernetes environment using the previous built [image](#the-development-docker-image).

For Minikube, run the following command to launch the job.
```bash
kubectl apply -f manifests/examples/elasticdl-demo-minikube.yaml
```
Note that, to let Minikube use local image instead of remote registries, you need to run `eval $(minikube docker-env)` first, and then build the image following [instructions](#the-development-docker-image).

For other Kubernetes clusters, first make sure the built image has been pushed to some registries, and then run the following command to launch the job. 
```bash
kubectl apply -f manifests/examples/elasticdl-demo-k8s.yaml
```

If you find permission error in the main pod log, e.g., `"pods is forbidden: User \"system:serviceaccount:default:default\" cannot create resource \"pods\""`, you need to grant pod-related permissions for the default user.
```bash
kubectl apply -f manifests/examples/elasticdl-rbac.yaml
```

### Run ElasticDL On GKE
https://docs.google.com/document/d/1cbzYjHTd7SUAPUGbT82AcmVu_4iWWjwykrryIDH-fOU/edit#heading=h.mbjsiz6n6jlo

### ElasticDL Priority-Based Elastic Scheduling
https://docs.google.com/document/d/1H4rhRc5Ll0uxTkiVf2fV_Q0xFf0HlKas1IAqPtWGHcQ/edit#heading=h.vb8p0lepu9vn (Note: The doc is currently in Chinese.)
