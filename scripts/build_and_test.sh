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

sh scripts/build.sh

#(
#    cd /tmp/elasticdl
#    go test -v -cover ./...
#)

# Run Python unittests

#pytest elasticdl/python/tests \
#    elasticdl_preprocessing/tests \
#    elasticdl_client/tests \
#    --cov=elasticdl/python \
#    --cov-report=xml
pytest elasticdl/python/tests/pserver_servicer_test.py::PserverServicerTest::test_pull_embedding_vectors
mkdir -p ./build
mv coverage.xml ./build
