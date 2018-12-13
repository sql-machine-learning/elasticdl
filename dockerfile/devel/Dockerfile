FROM tensorflow/tensorflow:1.12.0-devel-py3

# Code dependencies and dev tools
RUN add-apt-repository -y ppa:longsleep/golang-backports && apt-get update && apt-get -y --no-install-recommends install \
    libsnappy-dev \
    libgtest-dev \
    vim \
    bash-completion \
    less \
    iproute2 \
    golang-1.11 \
    tmux

RUN pip3 install \
    python-snappy \
    crc32c \
    torch torchvision

# So bazel won't complain about untracked header files
RUN ln -s /usr/local/lib/python3.5/dist-packages/tensorflow /usr/local/include 

COPY entry.sh /entry.sh
ENTRYPOINT ["/entry.sh"]
