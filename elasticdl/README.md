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
    --build-arg EXTRA_PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
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
make && python -m unittest discover elasticdl '*_test.py'
```

Could also start Docker container and run unittests in a single command:

```bash
docker run --rm -u $(id -u):$(id -g) -it \
    -v $EDL_REPO:/v \
    -w /v \
    elasticdl:dev \
    bash -c "make && python -m unittest discover elasticdl '*_test.py'"
```
### Test in Docker

In a terminal, start master to distribute mnist training tasks.

```
docker run --net=host --rm -it elasticdl:dev \
    python -m master.main \
        --model-file=examples/mnist/mnist.py \
        --model-class=MnistModel \
        --train_data_dir=/data/mnist/train \
        --record_per_task=100 \
        --num_epoch=2 \
        --grads_to_wait=2 \
        --minibatch_size=10
```

In another terminal, start a worker

```
docker run --net=host --rm -it elasticdl:dev \
    python -m worker.main \
        --model-file=examples/mnist/mnist.py \
        --model-class=MnistModel \
        --master_addr=localhost:50001
```

This will train MNIST data with a model defined in [examples/mnist/mnist.py](examples/mnist/mnist.py) for 2 epoches. 

### Test with Kubernetes
Create a `test_mnist.yaml` file as:

```
apiVersion: v1
kind: Pod
metadata:
  name: test-mnist
  labels:
    purpose: test-command
spec:
  containers:
  - name: mnist-demo-container
    image: edl:newtest
    command: ["python"]
    args: ["-m", "elasticdl.master.main", "--model-file=/elasticdl/examples/mnist/mnist.py", "--model-class=MnistModel", "--train_data_dir=/data/mnist/train", "--record_per_task=100", "--num_epoch=1", "--grads_to_wait=2", "--minibatch_size=64", "--num_worker=2", "--worker_image=elasticdl:dev", "--job_name=edl-train-043019"]
    imagePullPolicy: Never
    env:
      - name: MY_POD_IP
        valueFrom:
          fieldRef:
            fieldPath: status.podIP
  restartPolicy: Never
```

Then start the training job as:

```
kubectl apply -f test_mnist.yaml
```

This will start a master pod, which will launch 2 worker pods for training.

### Manual Debug

Sometimes it is easier to debug with a real master server. To start master server in container, run the following in `elasticdl` directory:

```bash
make && python -m master.main
```

### Run ElasticDL On GKE
https://docs.google.com/document/d/1cbzYjHTd7SUAPUGbT82AcmVu_4iWWjwykrryIDH-fOU/edit#heading=h.mbjsiz6n6jlo

### ElasticDL Priority-Based Elastic Scheduling
https://docs.google.com/document/d/1H4rhRc5Ll0uxTkiVf2fV_Q0xFf0HlKas1IAqPtWGHcQ/edit#heading=h.vb8p0lepu9vn (Note: The doc is currently in Chinese.)
