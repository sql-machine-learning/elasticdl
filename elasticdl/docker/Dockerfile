FROM tensorflow/tensorflow:1.13.1-py3
# For GPU version, use:
# FROM tensorflow/tensorflow:1.13.1-gpu-py3
# for pytorch, use:
# FROM pytorch/pytorch

RUN apt-get update
RUN apt-get install -y unzip curl

ARG EXTRA_PYPI_INDEX

# Install gRPC tools in Python
RUN pip install grpcio-tools --extra-index-url=${EXTRA_PYPI_INDEX}

# Install the Kubernetes Python client
RUN pip install kubernetes --extra-index-url=${EXTRA_PYPI_INDEX}

# Install Docker python SDK
RUN pip install docker --extra-index-url=${EXTRA_PYPI_INDEX}

# Install RecordIO dependencies
RUN apt-get install -y libsnappy-dev

# Install RecordIO 
RUN pip install 'pyrecordio==0.0.4' --extra-index-url=${EXTRA_PYPI_INDEX}

ENV PYTHONPATH=/:/elasticdl/python/

# ElasticDL dataset generator and codecs
COPY elasticdl/python/data /data  
RUN python /data/recordio_gen/mnist/gen_data.py --codec_type tf_example /data
RUN python /data/recordio_gen/cifar10/gen_data.py --codec_type tf_example /data

WORKDIR /
COPY elasticdl /elasticdl
COPY Makefile /Makefile
RUN make
