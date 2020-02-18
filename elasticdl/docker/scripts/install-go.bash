#!/bin/bash

set -e

GO_MIRROR_URL=$1

curl --silent ${GO_MIRROR_URL}/go1.13.4.linux-amd64.tar.gz | tar -C /usr/local -xzf -

go env -w GOPROXY=https://goproxy.io,direct

cp $GOPATH/bin/* /usr/local/bin/

mkdir -p $GOPATH/pkg/mod