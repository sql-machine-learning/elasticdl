#! /bin/bash
set -x
USER_ID=$(id -u)
GRP_ID=$(id -g)
USER_NAME=${USER}
if [[ $USER_ID == 0 ]]; then
    USER_ID=${SUDO_UID}
    GRP_ID=${SUDO_GID}
    USER_NAME=${SUDO_USER}
fi
HOME_DIR=$(eval echo ~${USER_NAME})

docker run --rm -it --net=host \
    -u ${USER_ID}:${GRP_ID} \
    -v ${HOME_DIR}:/home/${USER_NAME}  \
    -w /home/${USER_NAME} \
    -e USER=${USER_NAME} \
    -e HOME=/home/${USER_NAME} \
    reg.docker.alibaba-inc.com/elasticdl/dev /bin/bash
