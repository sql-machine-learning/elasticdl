import argparse
import unittest

from elasticdl.python.common.args import (
    add_bool_param,
    build_arguments_from_parsed_result,
    parse_ps_args,
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

    def test_parse_ps_args(self):
        num_ps_pods = 2
        minibatch_size = 16
        num_minibatches_per_task = 8
        model_zoo = "dummy_zoo"
        model_def = "dummy_def"

        original_args = [
            "--num_ps_pods",
            str(num_ps_pods),
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
        self.assertEqual(parsed_args.num_ps_pods, num_ps_pods)
        self.assertEqual(parsed_args.minibatch_size, minibatch_size)
        self.assertEqual(
            parsed_args.num_minibatches_per_task, num_minibatches_per_task
        )
        self.assertEqual(parsed_args.model_zoo, model_zoo)
        self.assertEqual(parsed_args.model_def, model_def)
