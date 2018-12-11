FROM tensorflow/tensorflow:1.12.0-devel-py3

# For recordio support
RUN apt-get update && apt-get -y --no-install-recommends install libsnappy-dev && pip3 install python-snappy

# So bazel won't complain about untracked header files
RUN ln -s /usr/local/lib/python3.5/dist-packages/tensorflow /usr/local/include 
