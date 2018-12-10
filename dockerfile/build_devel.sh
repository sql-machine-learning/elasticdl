#! /bin/bash
set -x
docker build \
  --build-arg USER=$USER \
  --build-arg UID=`id -u` \
  --build-arg GROUP=`id -g -n` \
  --build-arg GID=`id -g` \
  -t devel:$USER devel


