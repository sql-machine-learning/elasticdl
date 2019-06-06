ARG BASE_IMAGE=tensorflow/tensorflow:1.13.1-py3
FROM ${BASE_IMAGE}

RUN apt-get update
RUN apt-get install -y unzip curl

ARG EXTRA_PYPI_INDEX

# Install PySpark
RUN pip install pyspark --extra-index-url=${EXTRA_PYPI_INDEX}

# Install Pillow, which is needed to process sample data
RUN pip install Pillow --extra-index-url=${EXTRA_PYPI_INDEX}

# Install OpenJDK-8
RUN apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;
RUN apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

# Install gRPC tools in Python
RUN pip install grpcio-tools --extra-index-url=${EXTRA_PYPI_INDEX}

# Install RecordIO 
RUN pip install 'pyrecordio>=0.0.6' --extra-index-url=${EXTRA_PYPI_INDEX}

ENV PYTHONPATH=/
WORKDIR /
COPY elasticdl /elasticdl
COPY Makefile /Makefile
RUN make
ENTRYPOINT ["spark-submit"]
