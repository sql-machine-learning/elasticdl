FROM nvidia/cuda:9.2-base-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  clang \
  libcurl3 \
  libicu-dev \
  libpython-dev \
  libncurses5-dev \
  libxml2 \
  curl

WORKDIR /Swift4TF
# For latest version, check https://github.com/tensorflow/swift/blob/master/Installation.md
ARG VERSION="DEVELOPMENT-2019-01-04-a"
RUN curl -OLv https://storage.googleapis.com/swift-tensorflow/ubuntu16.04/swift-tensorflow-${VERSION}-ubuntu16.04.tar.gz && \
    tar xzvf swift-tensorflow-${VERSION}-ubuntu16.04.tar.gz && \
    rm swift-tensorflow-${VERSION}-ubuntu16.04.tar.gz

ENV PATH="/Swift4TF/usr/bin:${PATH}"
