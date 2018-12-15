I run this example using docker

```bash
docker run --rm -it -v $PWD:/work -w /work pytorch/pytorch bash
```

Then type `python mnist.py` in the container.  Or,

```bash
docker run --rm -it -v $PWD:/work -w /work pytorch/pytorch python mnist.py
```
