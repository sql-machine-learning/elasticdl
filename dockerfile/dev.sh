#! /bin/bash
set -x
GRP_ID=$(id -g)
USER_ID=$(id -u)
if [[ $USER_ID == 0 ]]; then
    USER_ID=${SUDO_UID}
    GRP_ID=${SUDO_GID}
fi

GRP_NAME=$(getent group ${GRP_ID} | cut -d: -f1)
USER_NAME=$(getent passwd ${USER_ID} | cut -d: -f1)
HOME_DIR=$(eval echo ~${USER_NAME})

docker run --rm -it --net=host \
    -v ${HOME_DIR}:/home/${USER_NAME}  \
    -e REAL_GID=${GRP_ID} \
    -e REAL_GRP=${GRP_NAME} \
    -e REAL_UID=${USER_ID} \
    -e REAL_USER=${USER_NAME} \
    reg.docker.alibaba-inc.com/elasticdl/dev /bin/bash
