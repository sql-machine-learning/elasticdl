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
    -v ${HOME_DIR}:/home/${USER_NAME}  \
    -e REAL_GID=${GRP_ID} \
    -e REAL_UID=${USER_ID} \
    -e REAL_USER=${USER_NAME} \
    reg.docker.alibaba-inc.com/elasticdl/dev /bin/bash
