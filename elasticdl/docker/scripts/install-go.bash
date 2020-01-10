#!/bin/bash

set -e

curl --silent https://dl.google.com/go/go1.13.4.linux-amd64.tar.gz | tar -C /usr/local -xzf -

go get github.com/golang/protobuf/protoc-gen-go
go get golang.org/x/lint/golint
go get golang.org/x/tools/cmd/goyacc
go get golang.org/x/tools/cmd/cover
go get github.com/mattn/goveralls
go get github.com/rakyll/gotest
go get github.com/wangkuiyi/goyaccfmt

cp $GOPATH/bin/* /usr/local/bin/
