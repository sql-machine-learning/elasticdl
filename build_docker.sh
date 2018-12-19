#! /bin/bash
cp -r dockerfile/user /tmp
cp -r python /tmp/user
cp launcher.py /tmp/user
docker build -t elasticdl/user /tmp/user 
