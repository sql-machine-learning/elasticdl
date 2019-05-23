#! /bin/bash
set -x

# delete the group with the given GID
grp_name=$(getent group ${REAL_GID} | cut -d: -f1)
if [[ ! -z ${grp_name} ]]; then 
    groupdel ${grp_name} || exit -1
fi

# delete the group with the given group name
groupdel ${REAL_GRP}

# delete the user with the given UID
usr_name=$(getent passwd ${REAL_USER} | cut -d: -f1)
if [[ ! -z ${user_name} ]]; then
    userdel ${user_name} || exit -1
fi

# delete the user with the given name
userdel ${REAL_USER}

# add group
groupadd -g ${REAL_GID} ${REAL_GRP} || exit -1 

# add user
useradd -r -g ${REAL_GID} -u ${REAL_UID} ${REAL_USER} -d /home/${REAL_USER} -s /bin/bash || exit -1

# Change root password to 'root'
echo -e "root\nroot" | (passwd root) || exit -2

su - ${REAL_USER}

