#! /bin/bash

useradd -r -g ${REAL_GID} -u ${REAL_UID} ${REAL_USER} -d ${REAL_HOME} -s /bin/bash || exit -1

su - ${REAL_USER}

