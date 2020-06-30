export BASE_IMAGE=tensorflow/tensorflow:2.1.0-py3

docker build --target dev -t elasticdl:dev -f elasticdl/docker/Dockerfile --build-arg BASE_IMAGE="$BASE_IMAGE" .

docker run --rm -it --net=host -v $PWD:/work -w /work elasticdl:dev bash -c "scripts/build.sh"
