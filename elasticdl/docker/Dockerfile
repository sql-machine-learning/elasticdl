ARG BASE_IMAGE=tensorflow/tensorflow:2.0.0b0-py3
ARG BUILD_ENV=data
FROM ${BASE_IMAGE} as base

RUN apt-get update && apt-get install -y unzip curl git

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

FROM base as build_data
ONBUILD COPY --from=elasticdl:data /data /data

FROM base as build_no_data
ONBUILD RUN echo "Skipped copying demo data"

FROM build_${BUILD_ENV}
ENV PYTHONPATH=/
WORKDIR /
COPY elasticdl /elasticdl
COPY elasticdl/Makefile /Makefile
RUN make

COPY elasticdl/docker/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc
