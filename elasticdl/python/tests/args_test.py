import argparse
import unittest

from elasticdl.python.common.args import (
    add_bool_param,
    build_arguments_from_parsed_result,
    remove_end_slash,
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

    def test_remove_end_slash(self):
        self.assertEqual("abc", remove_end_slash("abc/"))
        self.assertEqual("/abc", remove_end_slash("/abc/"))
        self.assertEqual("a/b/c", remove_end_slash("a/b/c/"))
        self.assertEqual("/a/b/c", remove_end_slash("/a/b/c/"))
        self.assertIsNone(remove_end_slash(None))
