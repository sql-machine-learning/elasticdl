I run this example using docker

```bash
docker run --rm -it -v $PWD:/work -w /work pytorch/pytorch bash
```

Then type `python mnist.py` in the container.  Or,

```bash
docker run --rm -it -v $PWD:/work -w /work pytorch/pytorch python mnist.py
```

After mnist script is complete, a png image with default name loss.png will be produced in the current directory which shows the curve of ps and all the trainer's loss varying with time. you also can specify the output png image name by running:

```bash
python mnist.py --loss-file ${image_name}.png
```

NOTE: if you are running the example on macOS, please make sure that you give the Docker engine sufficient amount of memory following [this guide](https://docs.docker.com/docker-for-mac/#advanced), or try to use a small number of trainer threads.  Otherwise, Docker might kill the process without giving any error message.
