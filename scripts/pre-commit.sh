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



PYTHON_SOURCE_FILES=$(find elasticdl/python \
                           elasticdl_preprocessing \
                           model_zoo \
                           tools \
                           setup.py \
                           scripts \
                           -name '*.py' -print0 | tr '\0' ' ')

GO_SOURCE_FILES=$(find elasticdl/pkg -name '*.go' -print0 | tr '\0' ' ')

YAML_SOURCE_FILES=$(find .travis.yml -name '*.yml' -print0 | tr '\0' ' ')

pre-commit run --files "$PYTHON_SOURCE_FILES $GO_SOURCE_FILES"
