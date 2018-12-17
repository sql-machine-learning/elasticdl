I run this example using docker

```bash
docker run --rm -it -v $PWD:/work -w /work pytorch/pytorch bash
```

Then type `python mnist.py` in the container.  Or,

```bash
docker run --rm -it -v $PWD:/work -w /work pytorch/pytorch python mnist.py
```

NOTE: if you are running the example on macOS, please make sure that you give the Docker engine sufficient amount of memory following [this guide](https://docs.docker.com/docker-for-mac/#advanced), or try to use a small number of trainer threads.  Otehrwise, Docker might kill the process without given any error message.
