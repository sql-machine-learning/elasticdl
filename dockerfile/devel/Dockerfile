FROM tensorflow/tensorflow:1.12.0-devel-py3

ARG GROUP
ARG GID
ARG USER
ARG UID

# For recordio support
RUN apt-get update && apt-get -y --no-install-recommends install libsnappy-dev && pip3 install python-snappy

# So bazel won't complain about untracked header files
RUN ln -s /usr/local/lib/python3.5/dist-packages/tensorflow /usr/local/include 

RUN groupdel $GROUP; groupdel `getent group $GID | cut -d: -f1`; groupadd -r -g $GID $GROUP && useradd -r -g $GID -u $UID $USER -s /bin/bash 

WORKDIR /home/$USER
USER $USER
 
