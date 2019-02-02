# Swift For Tensorflow

## Build S4TF Docker Image With GPU Support

We first build a devel image, the use the devel image to compile S4TF toolchain from source and finally add the compiled toolchain to build S4TF Docker Image.

### Build Devel Image

In `dockerfiles` directory: 
```
docker build \
    -t reg.docker.alibaba-inc.com/elasticdl/swift-gpu-devel \
    -t swift-gpu-devel \
    swift-gpu-devel
docker push reg.docker.alibaba-inc.com/elasticdl/swift-gpu-devel
```

### Compile S4TF Toolchain With GPU Support

On develop machine, follow the instructions here to clone the repo and prepate the many dependant repos.

```
git clone git@github.com:apple/swift.git -b tensorflow
./swift/utils/update-checkout --clone-with-ssh --scheme tensorflow
```

Now use devel image to build s4tf (You will need to mount your own git directories).

```
sudo docker run --net=host -it --rm \
    -v /home/l.zou/git:/git \
    reg.docker.alibaba-inc.com/elasticdl/swift-gpu-devel \ 
    /bin/bash -c \
    'cd /git/swift-1 && ./utils/build-toolchain-tensorflow -g'
```

（TDOO）currently bazel uses cache in the container, we could make it cache on a mounted directory to reuse across compilations.

If everything works ok, there will be a `swift-tensorflow-LOCAL-*.tar.gz` file generated in swift git directory.

### Build S4TF Image with the toolchain

Run `build-swift-gpu-image.sh` script and pass the path of the toolchain file to it. It will build a image tagged `s4tf-gpu`.

### Verify S4TF Image GPU Support

Start a container using `s4tf-gpu` image:

```
sudo docker run --net=host -it --rm \
    --security-opt seccomp:unconfined \
    s4tf-gpu /bin/bash
```

TODO: add steps for verification
