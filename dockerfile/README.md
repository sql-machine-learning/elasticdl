# Docker images

## Registry

We use AliCloud's Docker registry to host Docker images. Go to http://docker.alibaba-inc.com to register your account.

- Before you can use the registry in console, do `docker login --username=xxx reg.docker.alibaba-inc.com` to login to the registry.

## ElasticDL end user Docker image

The image is intended to be used to reproduce local experiment results and run on K8S (TODO: currently it is not useable on K8S yet.). A new image should be built locally when there is any code change in ElasticDL system.

### Build Image

To build image run the following script in elasticdl repo root:
```bash
./build_docker.sh
```
The docker image will be tagged `elasticdl/user`

### Use the image to train model on local machine

To train a model locally, you will need to mount a directory that contains user provided module and training data to the container and launch thread runner. Here is an example of training mnist model:
``` bash
docker run -it --rm -v $HOME:/work elasticdl/user \
   --class_name=MnistCNN \
   --runner=thread \
   --num_ps=2 \
   --num_worker=2 \
   --input=/work/data/mnist \
   /work/git/elasticdl/test/mnist.py
```
In the example, the user mount his home directory as the container's `/work` directory. User provided training data is at `$HOME/data/mnist` directory, and the user provided script is `$HOME/git/elasticdl/test/mnist.py`. The example script can be found [here](https://github.com/wangkuiyi/elasticdl/blob/develop/test/mnist.py)

### Use the image to train model in a K8S container
TODO

## ElasticDL develop Docker image

The image is intended to ease the developing process of ElasticDL. The Docker file is in `devel` directory. The pre-built image's registry entry is `reg.docker.alibaba-inc.com/elasticdl/dev`.

- Use the image for developing: run the script `dev.sh` or `sudo dev.sh` if you are not part of the `docker` system group. The script automatically determines your user id, group id from host machine, and propagate them to the docker container. This is to prevent the build process from creating artifacts under the root permission. The script also mounts your host home directory as the container home directory so you can build in the your git directories from there. Note that you might want to modify your `.bashrc` file to make it work on both host machine and in container.

- Build image: usually you don't need to build the image. Just use the one in the registry. But in case you need to modify the image to add more libraries, etc.
   ```bash
   docker build -t elasticdl/dev devel
   docker tag elasticdl/dev reg.docker.alibaba-inc.com/elasticdl/dev
   docker push reg.docker.alibaba-inc.com/elasticdl/dev
   ```
