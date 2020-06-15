#!/bin/bash
# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

WITH_K8S_TESTS=$1

# Build elasticdl.org/elasticdl Go package
go env -w GOPROXY=https://goproxy.cn,https://developer.aliyun.com/mirror/goproxy,direct
cd /elasticdl && \
make -f elasticdl/Makefile && \
cp -r /elasticdl/elasticdl/go/ /root/elasticdl && \
cd "$GOPATH"/pkg/mod/github.com/tensorflow && \
go mod init github.com/tensorflow &&\
cd /root/elasticdl && \
go mod init elasticdl.org/elasticdl && \
go mod edit -replace github.com/tensorflow="${GOPATH}"/pkg/mod/github.com/tensorflow && \
go get k8s.io/client-go@v0.17.0 && \
go mod tidy && \
go install ./...

# Run Go unittests
go test -v -cover ./...

# Run Python unittests
cd /elasticdl
K8S_TESTS=${WITH_K8S_TESTS} pytest elasticdl/python/tests elasticdl_preprocessing/tests --cov=elasticdl/python --cov-report=xml
mv coverage.xml build

# Create elasticdl package
mkdir /elasticdl/elasticdl/go/bin && cp /root/go/bin/elasticdl_ps /elasticdl/elasticdl/go/bin
cd /elasticdl && \
python setup.py --quiet bdist_wheel && \
cp dist/elasticdl-develop-py3-none-any.whl /elasticdl/build



