FROM tensorflow/tensorflow:1.13.1-py3
# For GPU version, use:
# FROM tensorflow/tensorflow:1.13.1-gpu-py3
# for pytorch, use:
# FROM pytorch/pytorch

RUN apt-get update
RUN apt-get install -y unzip curl

# Install gRPC tools in Python
RUN pip install grpcio-tools

# Install the Kubernetes Python client
RUN pip install kubernetes

# Install RecordIO and its dependency
RUN apt-get install -y libsnappy-dev && \
    pip install pyrecordio

RUN pip install http://dl-alipay-hz1.cn-hangzhou.oss.aliyun-inc.com/elasticdl-0.0.1-py3-none-any.whl && \
    python -m elasticdl.recordio_ds_gen.mnist.gen_data /data
