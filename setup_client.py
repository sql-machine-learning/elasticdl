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

with open("elasticdl_client/requirements.txt") as f:
    required_deps = f.read().splitlines()

setup(
    name="elasticdl_client",
    version="0.2.0rc1",
    description="The client tool for ElasticDL.",
    author="Ant Financial",
    url="https://elasticdl.org",
    install_requires=required_deps,
    python_requires=">=3.5",
    packages=find_packages(include=["elasticdl_client*"]),
    package_data={"": ["requirements.txt"]},
    entry_points={"console_scripts": ["elasticdl=elasticdl_client.main:main"]},
)
