# ElasticDL: A Kubernetes-native Deep Learning Framework

## The Development Docker Image

To build the development Docker image, run the following command:

```bash
docker build -t elasticdl:dev -f Dockerfile.dev .
```

To develop in the Docker container, run the following command to map in your git directory and start container:

```bash
docker run --rm -u $(id -u):$(id -g) -it -v $HOME/git:/git elasticdl:dev /bin/bash
```
