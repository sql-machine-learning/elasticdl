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

with open("elasticdl_preprocessing/requirements.txt") as f:
    required_deps = f.read().splitlines()

extras = {}
with open("elasticdl_preprocessing/requirements-dev.txt") as f:
    extras["develop"] = f.read().splitlines()

setup(
    name="elasticdl_preprocessing",
    version="0.2.0rc3.dev0",
    description="A feature preprocessing library.",
    long_description="This is an extension of the native Keras Preprocessing"
    " Layers and Feature Column API from TensorFlow. We can develop our model"
    " using the native high-level API from TensorFlow and our library."
    " We can train this model using native TensorFlow or ElasticDL.",
    long_description_content_type="text/markdown",
    author="Ant Financial",
    url="https://elasticdl.org",
    install_requires=required_deps,
    extras_require=extras,
    python_requires=">=3.5",
    packages=find_packages(
        include=["elasticdl_preprocessing*"], exclude=["*test*"]
    ),
    package_data={"": ["requirements.txt"]},
)
