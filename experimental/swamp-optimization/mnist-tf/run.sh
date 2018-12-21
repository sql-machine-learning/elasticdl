#! /bin/bash

#Edit the parameters below
#num_worker
W=1
#pull_probability
P=0.5
#evaluation_frequency
F=10
#log_image
PNG_IMAGE=accuracy_plot.png
#end of parameters editing

set -x
GRP_ID=$(id -g)
USER_ID=$(id -u)
if [[ $USER_ID == 0 ]]; then
    USER_ID=${SUDO_UID}
    GRP_ID=${SUDO_GID}
fi

if [[ $OSTYPE == "darwin"* ]]; then
    GRP_NAME=$(dscl . -search /Groups PrimaryGroupID ${GRP_ID} | head -1 | cut -f1)
    USER_NAME=$(dscl . -search /Users UniqueID ${USER_ID} | head -1 | cut -f1)
else
    GRP_NAME=$(getent group ${GRP_ID} | cut -d: -f1)
    USER_NAME=$(getent passwd ${USER_ID} | cut -d: -f1)
fi

HOME_DIR=$(eval echo ~${USER_NAME})

docker run --rm -it --hostname=elasticdl-dev --net=host \
    -v ${PWD}:/work -w /work  \
    -e REAL_GID=${GRP_ID} \
    -e REAL_GRP=${GRP_NAME} \
    -e REAL_UID=${USER_ID} \
    -e REAL_USER=${USER_NAME} \
    reg.docker.alibaba-inc.com/elasticdl/base  python src/swamp_launcher.py  \
    test/mnist.py --class_name MnistCNN --runner thread \
    --input /data/mnist   --num_worker ${W} --pull_probability ${P}  \
    --evaluation_frequency ${F} --log_image ${PNG_IMAGE}
