#!/bin/bash
set -e

make -f elasticdl/Makefile

# Go unittests
pushd $GOPATH/src/elasticdl.org/elasticdl
go install ./...
go test -v -cover ./...
popd

# Python unittests
K8S_TESTS=True pytest elasticdl/python/tests --cov=elasticdl/python --cov-report=xml
mv coverage.xml shared