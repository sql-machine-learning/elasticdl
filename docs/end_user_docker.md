# Docker image for end user
We are going to provide a Docker image to end user, who can run it locally or deploy it to K8S to launch ElasticDL system.

## iEnd user docker image usage
User can use command line like following to train a model.
```bash
docker run -it --rm -v $WORK_DIR:/work elasticdl:user python /elasticdl/launcher.py \
    --script=/work/mnist.py
    --class_name=MnistCNN
    --runner=thread
    --num_ps=2
    --num_worker=2
    --input=/data/mnist  
``` 

## Content in docker image
* tensorflow/tensorflow-1.12.0-py3
* pytorch 1.0.0
* ElasticDL's supporting python library
  * recordio
  * crc32c
  * snappy
* ElastiDL's python library
* Some prebuilt data in recordio format

## End user docker image build process
We are going to use `dockerfiles/elasticdl` directory to keep the image's dockerfile, which has the definition, install commands, etc.   The directory also has a copy of RecordIO wheel file.

Run `build_docker.sh` in the repo root directory, which will do the following:
1. copy `dockerfiles/elasticdl`to a staging directory, e.g. `/tmp/elasticdl`
1. copy preprocessed data to `/tmp/elasticdl`
1. copy all content of `python` directory to `/tmp/elasticdl`
1. run `docker build -t elasticdl:user /tmp/elasticdl` which builds the end user package.

