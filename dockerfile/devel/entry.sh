#! /bin/bash

useradd -r -g ${REAL_GID} -u ${REAL_UID} ${REAL_USER} -d /home/${REAL_USER} -s /bin/bash || exit -1

su - ${REAL_USER}

