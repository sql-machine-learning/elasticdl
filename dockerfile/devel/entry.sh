#! /bin/bash

useradd -r -g ${REAL_GID} -u ${REAL_UID} ${REAL_USER} -d /home/${REAL_USER} -s /bin/bash || exit -1

# Change root password to 'root'
echo -e "root\nroot" | (passwd root) || exit -2

su - ${REAL_USER}

