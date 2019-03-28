# ElasticDL: A Kubernetes-native Deep Learning Framework

## The Development Docker Image

To build the development Docker image, run the following command for pytorch version:
```bash
docker build -t elasticdl:dev -f Dockerfile.dev .
```

for tensorflow version:
```bash
docker build -t elasticdl:tfdev -f Dockerfile.tfdev .
```

