FROM tensorflow/tensorflow:1.12.0-py3

# Code dependencies and dev tools
RUN apt-get update && apt-get -y --no-install-recommends install \
    libsnappy-dev 

RUN pip3 install \
    python-snappy \
    crc32c \
    torch torchvision

COPY recordio-0.0.1-py3-none-any.whl /
RUN pip install /recordio-0.0.1-py3-none-any.whl

ADD python /elasticdl
ADD launcher.py /elasticdl

# TODO: copy prebuilt data
ENV PYTHONPATH /elasticdl/python
ENTRYPOINT ["python", "/elasticdl/launcher.py"]

