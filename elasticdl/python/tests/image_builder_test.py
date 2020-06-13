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

import os
import unittest

from elasticdl.python.elasticdl.image_builder import (
    _create_dockerfile,
    _find_elasticdl_root,
    _generate_unique_image_name,
)


class DockerTest(unittest.TestCase):
    def test_find_elasticdl_root(self):
        rdm = os.path.join(_find_elasticdl_root(), "README.md")
        self.assertTrue(os.path.exists(rdm))
        with open(rdm, "r") as f:
            self.assertTrue(f.read().startswith("# ElasticDL:"))

    def test_generate_unique_image_name(self):
        self.assertTrue(
            _generate_unique_image_name(None).startswith("elasticdl:")
        )
        self.assertTrue(
            _generate_unique_image_name("").startswith("elasticdl:")
        )
        self.assertTrue(
            _generate_unique_image_name("proj").startswith("proj/elasticdl:")
        )
        self.assertTrue(
            _generate_unique_image_name("gcr.io/proj").startswith(
                "gcr.io/proj/elasticdl:"
            )
        )

    def test_create_dockerfile(self):
        self.assertTrue(
            "COPY" in _create_dockerfile("elasticdl", "/home/me/models")
        )
        self.assertTrue(
            "COPY" in _create_dockerfile("elasticdl", "file:///home/me/models")
        )
        self.assertTrue(
            "git clone"
            in _create_dockerfile("elasticdl", "https://github.com/me/models")
        )
        with self.assertRaises(RuntimeError):
            _create_dockerfile("elasticdl", "")
        with self.assertRaises(RuntimeError):
            _create_dockerfile("elasticdl", None)
