FROM tensorflow/tensorflow:1.12.0-devel-py3

ARG GROUP
ARG GID
ARG USER
ARG UID

RUN groupdel $GROUP; groupdel `getent group $GID | cut -d: -f1`; groupadd -r -g $GID $GROUP && useradd -r -g $GID -u $UID $USER -s /bin/bash 
WORKDIR /home/$USER
USER $USER
 
