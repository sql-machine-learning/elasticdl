#!/bin/bash

set -e

GO_MIRROR_URL=$1

curl --silent ${GO_MIRROR_URL}/go1.13.4.linux-amd64.tar.gz | tar -C /usr/local -xzf -

go env -w GOPROXY=https://goproxy.io,direct

go get github.com/golang/protobuf/protoc-gen-go
go get golang.org/x/lint/golint
go get golang.org/x/tools/cmd/goyacc
go get golang.org/x/tools/cmd/cover
go get github.com/mattn/goveralls
go get github.com/rakyll/gotest
go get github.com/wangkuiyi/goyaccfmt
go get github.com/stretchr/testify/assert

cp $GOPATH/bin/* /usr/local/bin/
