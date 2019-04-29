# ElasticDL: A Kubernetes-native Deep Learning Framework

## The Development Docker Image

To build the development Docker image, run the following command:

```bash
docker build -t elasticdl:dev -f Dockerfile.dev .
```

To develop in the Docker container, run the following command to map in your `elasticdl` git repo directory and start container:

```bash
docker run --rm -u $(id -u):$(id -g) -it \
    -v $HOME/git/elasticdl:/elasticdl \
    elasticdl:dev
```

## Test and Debug

### Unittests

In dev Docker container's `elasticdl` repo's root directory, do the following:

```bash
make && python -m unittest elasticdl/*/*_test.py
```

Could also start Docker container and run unittests in a single command:

```bash
docker run --rm -u $(id -u):$(id -g) -it \
    -v $HOME/git/elasticdl:/elasticdl \
    -w /elasticdl/elasticdl \
    elasticdl:dev \
    bash -c "make && python -m unittest -v */*_test.py"
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

### Manual Debug

Sometimes it is easier to debug with a real master server. To start master server in container, run the following in `elasticdl` directory:

```bash
make && python -m master.main
```
