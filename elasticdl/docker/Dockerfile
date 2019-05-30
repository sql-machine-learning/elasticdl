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

# Install pytest
RUN pip install pytest --extra-index-url=${EXTRA_PYPI_INDEX}

# Install RecordIO 
RUN pip install 'pyrecordio>=0.0.6' --extra-index-url=${EXTRA_PYPI_INDEX}

ENV PYTHONPATH=/

WORKDIR /
COPY elasticdl /elasticdl
COPY Makefile /Makefile
RUN make
