#!/bin/bash
set -e

make -f elasticdl/Makefile

# Go unittests
rm -rf $GOPATH/src/elasticdl.org/elasticdl
mkdir -p $GOPATH/src/elasticdl.org/elasticdl
cp -r elasticdl/pkg $GOPATH/src/elasticdl.org/elasticdl
pushd $GOPATH/src/elasticdl.org/elasticdl
go get ./...
go install ./...
go test ./...
popd

# Python unittests
K8S_TESTS=True pytest elasticdl/python/tests --cov=elasticdl/python --cov-report=xml
mv coverage.xml shared