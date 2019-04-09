# ElasticDL: A Kubernetes-native Deep Learning Framework

## The Development Docker Image

To build the development Docker image, run the following command:

```bash
docker build -t elasticdl:dev -f Dockerfile.dev whl_file_dir
```
`whl_file_dir` is the directory which includes `recordio-0.0.1-py3-none-any.whl`. `recordio-0.0.1-py3-none-any.whl` can be downloaded from [here](https://github.com/ElasticDL/pyrecordio/raw/develop/artifacts/recordio-0.0.1-py3-none-any.whl).

To develop in the Docker container, run the following command to map in your `elasticdl` git repo directory and start container:

```bash
docker run --rm -u $(id -u):$(id -g) -it \
    -v $HOME/git/elasticdl:/elasticdl \
    elasticdl:dev
```

## Test and Debug

### Unittests

In dev Docker container's `elasticdl` repo's `elasticdl` directory, do the following:

```bash
make && python -m unittest -v */*_test.py
```

Could also start Docker container and run unittests in a single command:

```bash
docker run --rm -u $(id -u):$(id -g) -it \
    -v $HOME/git/elasticdl:/elasticdl \
    -w /elasticdl/elasticdl \
    elasticdl:dev \
    bash -c "make && python -m unittest -v */*_test.py"
```

### Manual Debug

Sometimes it is easier to debug with a real master server. To start master server in container, run the following in `elasticdl` directory:

```bash
make && python -m master.main
```
