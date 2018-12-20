#!!!!!!!!
# Dont use the Dockerfile directly. Use build.sh instead, otherwise the data section will be missing!
#!!!!!!!!

FROM tensorflow/tensorflow:1.12.0-py3

# ElasticDL Code dependencies
RUN apt-get update && apt-get -y --no-install-recommends install \
    libsnappy-dev 

RUN pip3 install \
    python-snappy \
    crc32c \
    torch torchvision

RUN pip3 install -I https://github.com/ElasticDL/pyrecordio/tree/develop/artifacts/recordio-0.0.1-py3-none-any.whl

# ElasticDL prebuilt datasets
COPY data /data

