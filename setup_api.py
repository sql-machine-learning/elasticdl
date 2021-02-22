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

from setuptools import find_packages, setup

with open("elasticai_api/requirements.txt") as f:
    required_deps = f.read().splitlines()

tensorflow_require_list = ["tensorflow"]
pytorch_require_list = ["torch"]

setup(
    name="elasticai-api",
    version="0.3.0rc0.dev0",
    description="The model development api for ElasticDL.",
    long_description="This is the sdk for developing ElasticDL models."
    " Model developers can use these APIs to support elastic and"
    " fault-tolerant training for their TensorFlow or Pytorch models.",
    long_description_content_type="text/markdown",
    author="Ant Group",
    url="https://elasticdl.org",
    python_requires=">=3.5",
    packages=find_packages(include=["elasticai_api*"], exclude=["*test*"]),
    install_requires=required_deps,
    extras_require={
        "tensorflow": tensorflow_require_list,
        "pytorch": pytorch_require_list,
    },
)
