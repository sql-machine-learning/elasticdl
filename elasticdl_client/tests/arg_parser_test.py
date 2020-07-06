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

import unittest

from elasticdl_client.common.args import DEFAULT_BASE_IMAGE
from elasticdl_client.main import build_argument_parser


class ArgParserTest(unittest.TestCase):
    def setUp(self):
        self._parser = build_argument_parser()

    def test_parse_zoo_init(self):
        args = ["zoo", "init"]
        args = self._parser.parse_args(args)
        self.assertEqual(args.base_image, DEFAULT_BASE_IMAGE)
        args.func(args)

        args = ["zoo", "init", "--base_image=elasticdl:base"]
        args = self._parser.parse_args(args)
        self.assertEqual(args.base_image, "elasticdl:base")

        with self.assertRaises(SystemExit):
            args = ["zoo", "init", "--mock_param=mock_value"]
            args = self._parser.parse_args(args)

    def test_parse_zoo_build(self):
        args = [
            "zoo",
            "build",
            "--image=a_docker_registry/bright/elasticdl-wnd:1.0",
            ".",
        ]
        args = self._parser.parse_args(args)
        self.assertEqual(
            args.image, "a_docker_registry/bright/elasticdl-wnd:1.0"
        )
        self.assertEqual(args.path, ".")

        with self.assertRaises(SystemExit):
            args = ["zoo", "build", "."]
            args = self._parser.parse_args(args)

        with self.assertRaises(SystemExit):
            args = [
                "zoo",
                "build",
                "--image=a_docker_registry/bright/elasticdl-wnd:1.0",
            ]
            args = self._parser.parse_args(args)

    def test_parse_zoo_push(self):
        args = ["zoo", "push", "a_docker_registry/bright/elasticdl-wnd:1.0"]
        args = self._parser.parse_args(args)
        self.assertEqual(
            args.image, "a_docker_registry/bright/elasticdl-wnd:1.0"
        )

        with self.assertRaises(SystemExit):
            args = ["zoo", "push"]
            args = self._parser.parse_args(args)
