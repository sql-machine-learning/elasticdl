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

# Some of the code here is copied from
# https://github.com/canha/golang-tools-install-script/blob/master/goinstall.sh

# Install Golang, which is needed to install pyrecordio
VERSION="1.12.5"

[ -z "$GOROOT" ] && GOROOT="$HOME/.go"
[ -z "$GOPATH" ] && GOPATH="$HOME/go"

OS="$(uname -s)"
ARCH="$(uname -m)"

case $OS in
  "Linux")
    case $ARCH in
      "x86_64")
        ARCH=amd64
      ;;
      "armv6")
        ARCH=armv6l
      ;;
      "armv8")
        ARCH=arm64
      ;;
      .*386.*)
        ARCH=386
      ;;
    esac
    PLATFORM="linux-$ARCH"
  ;;
  "Darwin")
    PLATFORM="darwin-amd64"
  ;;
esac

PACKAGE_NAME="go$VERSION.$PLATFORM.tar.gz"

if [ -d "$HOME/.go" ]; then
  echo "The '.go' directory already exists. Exiting."
  exit 1
fi

echo "Downloading $PACKAGE_NAME ..."
if hash wget 2>/dev/null; then
  wget https://storage.googleapis.com/golang/$PACKAGE_NAME -O /tmp/go.tar.gz
else
  curl -o /tmp/go.tar.gz https://storage.googleapis.com/golang/$PACKAGE_NAME
fi

echo "Extracting File..."
mkdir -p "$HOME/.go/"
tar -C "$HOME/.go" --strip-components=1 -xzf /tmp/go.tar.gz
cat >> "$HOME/.bashrc" <<EOF
export GOROOT=$GOROOT
export PATH=$PATH:$GOROOT/bin
export GOPATH=$GOPATH
export PATH=$PATH:$GOPATH/bin
EOF

mkdir -p "$GOPATH"/{src,pkg,bin}
rm -f /tmp/go.tar.gz

# Install pip
easy_install pip

# Install the dependencies we need
pip install --user pyrecordio>=0.0.6 Pillow

# A hacky fix for tensorflow installation
rm -rf /opt/conda/default/lib/python3.6/site-packages/wrapt*

# Install tensorflow
pip install tensorflow
