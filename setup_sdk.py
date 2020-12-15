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

tensorflow_require_list = ["tensorflow"]
pytorch_require_list = ["torch"]

setup(
    name="elasticdl_sdk",
    version="0.2.0rc3.dev0",
    description="The model development sdk for ElasticDL.",
    long_description="This is the sdk for developing ElasticDL models."
    " Model developers can make slight update using the sdk based on their"
    " TensorFlow or Pytorch models. And then the model can be trained in the"
    " elastic and fault-tolerant way.",
    long_description_content_type="text/markdown",
    author="Ant Financial",
    url="https://elasticdl.org",
    python_requires=">=3.5",
    packages=find_packages(include=["elasticdl_sdk*"], exclude=["*test*"]),
    extras_require={
        "tensorflow": tensorflow_require_list,
        "pytorch": pytorch_require_list,
    },
)
