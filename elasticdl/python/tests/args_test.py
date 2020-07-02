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

from elasticdl.python.common.args import (
    parse_ps_args,
    wrap_go_args_with_string,
)


class ArgsTest(unittest.TestCase):
    def test_parse_ps_args(self):
        minibatch_size = 16
        num_minibatches_per_task = 8
        model_zoo = "dummy_zoo"
        model_def = "dummy_def"

        original_args = [
            "--ps_id",
            str(0),
            "--port",
            str(2222),
            "--minibatch_size",
            str(minibatch_size),
            "--model_zoo",
            model_zoo,
            "--model_def",
            model_def,
            "--job_name",
            "test _args",
            "--num_minibatches_per_task",
            str(num_minibatches_per_task),
        ]
        parsed_args = parse_ps_args(original_args)
        self.assertEqual(parsed_args.ps_id, 0)
        self.assertEqual(parsed_args.port, 2222)
        self.assertEqual(parsed_args.minibatch_size, minibatch_size)
        self.assertEqual(
            parsed_args.num_minibatches_per_task, num_minibatches_per_task
        )
        self.assertEqual(parsed_args.model_zoo, model_zoo)
        self.assertEqual(parsed_args.model_def, model_def)

    def test_wrap_go_args_with_string(self):
        args = [
            "-ps_id=0",
            "-job_name=test_args",
            "-opt_args=learning_rate=0.1;momentum=0.0;nesterov=False",
        ]
        args = wrap_go_args_with_string(args)
        expected_args = [
            "-ps_id='0'",
            "-job_name='test_args'",
            "-opt_args='learning_rate=0.1;momentum=0.0;nesterov=False'",
        ]
        self.assertListEqual(args, expected_args)
