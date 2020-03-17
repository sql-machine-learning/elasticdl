#!/bin/bash
set -e

make -f elasticdl/Makefile

# Go unittests
pushd $ELASTICDLPATH
go test -v -cover ./...
popd

# Python unittests
K8S_TESTS=True pytest elasticdl/python/tests elasticdl_preprocessing/tests --cov=elasticdl/python --cov-report=xml
mv coverage.xml shared