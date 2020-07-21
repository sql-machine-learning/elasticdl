# ElasticDL: Developer's Guide

## Get the Source Code

You can get the latest source code from Github:

```bash
git clone https://github.com/sql-machine-learning/elasticdl
cd elasticdl
```

## Development Tools in a Docker Image

We prefer to install all building tools in a Docker image.

```bash
scripts/travis/build_images.sh
```

## Check Code Style

The above Docker image contains pre-commit and hooks.  We can run it as a
container and mount the local Git repo into the container.  In this
container, we run the pre-commit command.

```bash
docker run --rm -it -v $PWD:/work -w /work elasticdl:dev \
  bash -c "make -f elasticdl/Makefile && pre-commit run -a"
```

If you have all required tools installed, you can also run the same script
on your host.

```bash
make -f elasticdl/Makefile && pre-commit run -a
```

## Build Wheel Packages Using Your Modified Code

After modifying code, you can build wheel packages by running the
following command in the root of the project.

```bash
scripts/docker_build_wheel.sh
```

If you have `elasticdl:dev` image, you can only run the following command to
generate wheel packages in the root of the project.

```bash
docker run --rm -it --net=host -v "$PWD":/work -w /work elasticdl:dev \
    bash -c "scripts/build.sh"
```

You can find wheel packages in the `build` directory.

## Build a Docker Image using Local Wheel Packages

After building wheel packages, you can build a docker image by
`elasticdl zoo`.

```bash
elasticdl zoo init \
  --base_image=elasticdl:dev \
  --model_zoo=model_zoo \
  --local_pkg_dir=./build \

elasticdl zoo build --image_name=elasticdl:dev_test .
```

Then, you can submit a job using the image like the
[tutorial](https://github.com/sql-machine-learning/elasticdl/blob/develop/docs/tutorials/elasticdl_local.md).
