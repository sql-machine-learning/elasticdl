import unittest

from elasticdl.python.common.k8s_resource import parse


class K8SResourceTest(unittest.TestCase):
    def test_k8s_resource_parse(self):
        # parse works as expected on the allowed list of resources
        self.assertEqual(
            {
                "cpu": "250m",
                "memory": "32Mi",
                "disk": "64Mi",
                "nvidia.com/gpu": "1",
                "ephemeral-storage": "32Mi",
            },
            parse(
                "cpu=250m,memory=32Mi,disk=64Mi,gpu=1,ephemeral-storage=32Mi"
            ),
        )
        # When cpu is non-numeric, parse works as expected
        self.assertEqual(
            {
                "cpu": "250m",
                "memory": "32Mi",
                "disk": "64Mi",
                "nvidia.com/gpu": "1",
            },
            parse("cpu=250m,memory=32Mi,disk=64Mi,gpu=1"),
        )
        # When cpu is integer, parse works as expected
        self.assertEqual(
            {
                "cpu": "1",
                "memory": "32Mi",
                "disk": "64Mi",
                "nvidia.com/gpu": "1",
            },
            parse("cpu=1,memory=32Mi,disk=64Mi,gpu=1"),
        )
        # When cpu is float, parse works as expected
        self.assertEqual(
            {
                "cpu": "0.1",
                "memory": "32Mi",
                "disk": "64Mi",
                "nvidia.com/gpu": "1",
            },
            parse("cpu=0.1,memory=32Mi,disk=64Mi,gpu=1"),
        )
        # When cpu is non-numeric, raise an error
        self.assertRaisesRegex(
            ValueError,
            "invalid cpu request spec: 250Mi",
            parse,
            "cpu=250Mi,memory=32Mi,disk=64Mi,gpu=1",
        )
        # When gpu is non-numeric, raise an error
        self.assertRaisesRegex(
            ValueError,
            "invalid gpu request spec: 1Mi",
            parse,
            "cpu=2,memory=32Mi,disk=64Mi,gpu=1Mi",
        )
        # When gpu is not integer, raise an error
        self.assertRaisesRegex(
            ValueError,
            "invalid gpu request spec: 0.1",
            parse,
            "cpu=2,memory=32Mi,disk=64Mi,gpu=0.1",
        )
        # When gpu resource name has a valid vendor name,
        # parse works as expected
        self.assertEqual(
            {
                "cpu": "0.1",
                "memory": "32Mi",
                "disk": "64Mi",
                "amd.com/gpu": "1",
            },
            parse("cpu=0.1,memory=32Mi,disk=64Mi,amd.com/gpu=1"),
        )
        # When gpu resource name does not have a valid vendor name,
        # raise an error
        self.assertRaisesRegex(
            ValueError,
            "gpu resource name does not have a valid vendor name: blah-gpu",
            parse,
            "cpu=2,memory=32Mi,disk=64Mi,blah-gpu=1",
        )
        # When gpu resource name does not have a valid vendor name,
        # raise an error
        self.assertRaisesRegex(
            ValueError,
            "gpu resource name does not have a valid vendor name: @#/gpu",
            parse,
            "cpu=2,memory=32Mi,disk=64Mi,@#/gpu=1",
        )
        self.assertRaisesRegex(
            ValueError,
            "gpu resource name does not have a valid vendor name: a_d.com/gpu",
            parse,
            "cpu=2,memory=32Mi,disk=64Mi,a_d.com/gpu=1",
        )
        self.assertRaisesRegex(
            ValueError,
            "gpu resource name does not have a valid vendor name: *",
            parse,
            "cpu=2,memory=32Mi,disk=64Mi," + "a" * 64 + "/gpu=1",
        )
        # When memory does not contain expected regex, raise an error
        self.assertRaisesRegex(
            ValueError,
            "invalid memory request spec: 32blah",
            parse,
            "cpu=250m,memory=32blah,disk=64Mi,gpu=1",
        )
        # When resource name is unknown, raise an error
        self.assertRaisesRegex(
            ValueError,
            "unknown is not in the allowed list of resource types:",
            parse,
            "cpu=250m,unknown=32Mi,disk=64Mi,gpu=1",
        )
        # When resource name is duplicated, raise an error
        self.assertRaisesRegex(
            ValueError,
            "The resource string contains duplicate resource names: cpu",
            parse,
            "cpu=250m,cpu=32Mi,disk=64Mi,gpu=1",
        )
