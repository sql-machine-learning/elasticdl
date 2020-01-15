#!/bin/bash
set -e

make -f elasticdl/Makefile

# Go unittests
cd elasticdl
go install ./...
go test ./...
cd ..

# Python unittests
K8S_TESTS=True pytest elasticdl/python/tests --cov=elasticdl/python --cov-report=xml
mv coverage.xml shared