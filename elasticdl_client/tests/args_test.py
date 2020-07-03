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

import argparse
import unittest

from elasticdl_client.common.args import (
    add_bool_param,
    build_arguments_from_parsed_result,
    wrap_python_args_with_string,
)


class ArgsTest(unittest.TestCase):
    def setUp(self):
        self._parser = argparse.ArgumentParser()
        self._parser.add_argument("--foo", default=3, type=int)
        add_bool_param(self._parser, "--bar", False, "")

    def test_build_arguments_from_parsed_result(self):
        args = ["--foo", "4", "--bar"]
        results = self._parser.parse_args(args=args)
        original_arguments = build_arguments_from_parsed_result(results)
        value = "\t".join(sorted(original_arguments))
        target = "\t".join(sorted(["--foo", "4", "--bar", "True"]))
        self.assertEqual(value, target)

        original_arguments = build_arguments_from_parsed_result(
            results, filter_args=["foo"]
        )
        value = "\t".join(sorted(original_arguments))
        target = "\t".join(sorted(["--bar", "True"]))
        self.assertEqual(value, target)

        args = ["--foo", "4"]
        results = self._parser.parse_args(args=args)
        original_arguments = build_arguments_from_parsed_result(results)
        value = "\t".join(sorted(original_arguments))
        target = "\t".join(sorted(["--bar", "False", "--foo", "4"]))
        self.assertEqual(value, target)

    def test_wrap_python_args_with_string(self):
        args = [
            "--ps_id",
            str(0),
            "--job_name",
            "test_args",
            "--checkpoint_dir",
            "",
        ]
        args = wrap_python_args_with_string(args)
        expected_args = [
            "--ps_id",
            "'0'",
            "--job_name",
            "'test_args'",
            "--checkpoint_dir",
            "''",
        ]
        self.assertListEqual(args, expected_args)
