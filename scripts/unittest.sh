#!/bin/bash
set -e

make -f elasticdl/Makefile

# Go unittests
rm -rf $GOPATH/src/elasticdl.org/elasticdl
mkdir -p $GOPATH/src/elasticdl.org/elasticdl
cp -r elasticdl/pkg $GOPATH/src/elasticdl.org/elasticdl
pushd $GOPATH/src/elasticdl.org/elasticdl
export GO111MODULE=on
go mod init elasticdl.org/elasticdl
go install ./...
go test ./...
popd

# Python unittests
K8S_TESTS=True pytest elasticdl/python/tests --cov=elasticdl/python --cov-report=xml
mv coverage.xml shared