make -f elasticdl/Makefile

# Python unittests
K8S_TESTS=True pytest elasticdl/python/tests --cov=elasticdl/python --cov-report=xml
mv coverage.xml shared


# Go unittests
go install ./...
go test ./...