#! /bin/bash
set -e
set -x

tmp_dir=$(mktemp -d /tmp)
cp Dockerfile $tmp_dir

