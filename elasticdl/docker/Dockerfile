ARG BASE_IMAGE=tensorflow/tensorflow:1.13.1-py3
FROM ${BASE_IMAGE}

RUN apt-get update
RUN apt-get install -y unzip curl git

ARG EXTRA_PYPI_INDEX

# Install gRPC tools in Python
RUN pip install grpcio-tools --extra-index-url=${EXTRA_PYPI_INDEX}

# Install the Kubernetes Python client
RUN pip install kubernetes --extra-index-url=${EXTRA_PYPI_INDEX}

# Install Docker python SDK
RUN pip install docker --extra-index-url=${EXTRA_PYPI_INDEX}

# Install pytest
RUN pip install pytest --extra-index-url=${EXTRA_PYPI_INDEX}

# Install RecordIO 
RUN pip install 'pyrecordio>=0.0.6' --extra-index-url=${EXTRA_PYPI_INDEX}

# Dependencies for pre-commit hooks
RUN pip install pre-commit --extra-index-url=${EXTRA_PYPI_INDEX}

# Install Pillow for sample data processing Spark job
RUN pip install Pillow --extra-index-url=${EXTRA_PYPI_INDEX}

ENV PYTHONPATH=/

WORKDIR /
COPY elasticdl /elasticdl

# ElasticDL dataset generator and codecs
ARG GEN_DATA="no"
RUN if [ "$GEN_DATA" = "yes" ]; then \
        python elasticdl/python/data/recordio_gen/mnist/gen_data.py --codec_type tf_example elasticdl/python/data && \
        python elasticdl/python/data/recordio_gen/cifar10/gen_data.py --codec_type tf_example elasticdl/python/data; \
    else echo "Data generation skipped."; fi

COPY Makefile /Makefile
RUN make
