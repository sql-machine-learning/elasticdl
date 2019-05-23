# Swift For Tensorflow

## Build S4TF Docker Image With GPU Support

We first build a devel image, the use the devel image to compile S4TF toolchain from source and finally add the compiled toolchain to build S4TF Docker Image.

### Build Devel Image

In `dockerfiles` directory: 
```
docker build \
    -t swift-gpu-devel \
    swift-gpu-devel
```

### Compile S4TF Toolchain With GPU Support

On develop machine, follow the instructions here to clone the repo and prepate the many dependant repos.

```
git clone git@github.com:apple/swift.git -b tensorflow
./swift/utils/update-checkout --clone-with-ssh --scheme tensorflow
```

Now use devel image to build s4tf (You will need to mount your own git directories).

```
sudo docker run -it --rm \
    -v $HOME/.cache:/root/.cache \
    -v $HOME/git:/git \
    swift-gpu-devel \ 
    /bin/bash -c \
    'cd /git/swift && ./utils/build-toolchain-tensorflow -g'
```

Here `$HOME/.cache` volume is used for bazel cache across compilations.

If everything works ok, there will be a `swift-tensorflow-LOCAL-*.tar.gz` file generated in swift git directory.

### Build S4TF Image with the toolchain

Run `build-s4tf-image.sh` script and pass the path of the toolchain file to it. It will build a image tagged `s4tf-gpu`.

### Verify S4TF Image GPU Support

Start a container using `s4tf-gpu` image:

```
sudo docker run --net=host -it --rm \
    --security-opt seccomp:unconfined \
    s4tf-gpu /bin/bash
```

We need `--security-opt seccomp:unconfined` here for swift interpreter to work. According to [this note]( https://github.com/zachgrayio/swift-tensorflow/blob/5f29d1cfae6f93c424e677ee21ab35ea245ff41a/README.md#run-a-repl):

> when running this interactive container with the standard -it, we also must run without the default seccomp profile with --security-opt seccomp:unconfined to allow the Swift REPL access to ptrace and run correctly.


TODO: add steps for verification
