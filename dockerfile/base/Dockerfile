#!!!!!!!!
# Dont use the Dockerfile directly. In this directory run ./build.sh, which
# will download MNIST and Fashion-MNIST data, process them and populate the
# data directory.
#!!!!!!!!

FROM tensorflow/tensorflow:1.12.0-py3

# ElasticDL Code dependencies
RUN apt-get update && apt-get -y --no-install-recommends install \
    libsnappy-dev 

RUN pip3 install \
    python-snappy \
    crc32c \
    torch torchvision

# Install pre-build RecordIO package
RUN pip3 install -I https://github.com/ElasticDL/pyrecordio/raw/develop/artifacts/recordio-0.0.1-py3-none-any.whl

# ElasticDL prebuilt datasets
COPY data /data
