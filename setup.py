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

with open("elasticdl/requirements.txt") as f:
    required_deps = f.read().splitlines()
required_deps.append("elasticdl_preprocessing")

extras = {}
with open("elasticdl/requirements-dev.txt") as f:
    extras["develop"] = f.read().splitlines()

setup(
    name="elasticdl",
    version="0.2.0rc1",
    description="A Kubernetes-native Deep Learning Framework",
    long_description="ElasticDL is a Kubernetes-native deep learning framework"
    " built on top of TensorFlow 2.0 that supports"
    " fault-tolerance and elastic scheduling.",
    long_description_content_type="text/markdown",
    author="Ant Financial",
    url="https://elasticdl.org",
    install_requires=required_deps,
    extras_require=extras,
    python_requires=">=3.5",
    packages=find_packages(
        exclude=[
            "*test*",
            "elasticdl_client*",
            "elasticdl_preprocessing*",
            "model_zoo*",
        ]
    ),
    package_data={
        "": [
            "proto/*.proto",
            "docker/*",
            "Makefile",
            "requirements.txt",
            "go/bin/elasticdl_ps",
            "go/pkg/kernel/capi/*",
        ]
    },
)
