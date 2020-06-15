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

# Build elasticdl.org/elasticdl Go package
GO_PROXY_CN=https://goproxy.cn
GO_PROXY_ALIYUN=https://developer.aliyun.com/mirror/goproxy
go env -w GOPROXY="$GO_PROXY_CN","$GO_PROXY_ALIYUN",direct

# NOTE: The following commands requires that the current working directory is
# the source tree root.
make -f elasticdl/Makefile

# TODO(QiJune): How about we put the go.mod file in elasticdl/elasticdl/go/ in
# the Git repo, so we don't need to create it at building time?
(
    cp -r elasticdl/go/ /tmp/elasticdl
    cd "$GOPATH"/pkg/mod/github.com/tensorflow
    go mod init github.com/tensorflow
    cd /tmp/elasticdl
    go mod init elasticdl.org/elasticdl
    go mod edit -replace github.com/tensorflow="${GOPATH}"/pkg/mod/github.com/tensorflow
    go get k8s.io/client-go@v0.17.0
    go mod tidy
    go install ./...
    go test -v -cover ./...
)

# Run Python unittests
pytest elasticdl/python/tests elasticdl_preprocessing/tests --cov=elasticdl/python --cov-report=xml
mkdir -p ./build
mv coverage.xml /elasticdl/build

# Create elasticdl package
mkdir ./elasticdl/go/bin
cp "$GOPATH"/go/bin/elasticdl_ps ./elasticdl/go/bin
python setup.py --quiet bdist_wheel --dist-dir ./build
# cp dist/elasticdl-develop-py3-none-any.whl /elasticdl/build
