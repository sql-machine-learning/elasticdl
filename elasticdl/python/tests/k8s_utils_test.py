import unittest

from elasticdl.python.elasticdl.common.k8s_utils import parse_resource


class K8SUtilsTest(unittest.TestCase):

    def test_parse_resource(self):
        # parse_resource works as expected on the allowed list of resources
        self.assertEqual(
            {
                'cpu': '250m',
                'memory': '32Mi',
                'disk': '64Mi',
                'gpu': '1',
                'ephemeral-storage': '32Mi',
            },
            parse_resource(
                'cpu=250m,memory=32Mi,disk=64Mi,gpu=1,ephemeral-storage=32Mi'
            )
        )
        # When cpu is non-numeric, parse_resource works as expected
        self.assertEqual(
            {
                'cpu': '250m',
                'memory': '32Mi',
                'disk': '64Mi',
                'gpu': '1',
            },
            parse_resource(
                'cpu=250m,memory=32Mi,disk=64Mi,gpu=1'
            )
        )
        # When cpu is integer, parse_resource works as expected
        self.assertEqual(
            {
                'cpu': '1',
                'memory': '32Mi',
                'disk': '64Mi',
                'gpu': '1',
            },
            parse_resource(
                'cpu=1,memory=32Mi,disk=64Mi,gpu=1'
            )
        )
        # When cpu is float, parse_resource works as expected
        self.assertEqual(
            {
                'cpu': '0.1',
                'memory': '32Mi',
                'disk': '64Mi',
                'gpu': '1',
            },
            parse_resource(
                'cpu=0.1,memory=32Mi,disk=64Mi,gpu=1'
            )
        )
        # When cpu is non-numeric, raise an error
        self.assertRaisesRegex(
            ValueError,
            'invalid cpu request spec: 250Mi',
            parse_resource,
            'cpu=250Mi,memory=32Mi,disk=64Mi,gpu=1'
        )
        # When gpu is non-numeric, raise an error
        self.assertRaisesRegex(
            ValueError,
            'invalid gpu request spec: 1Mi',
            parse_resource,
            'cpu=2,memory=32Mi,disk=64Mi,gpu=1Mi'
        )
        # When gpu is not integer, raise an error
        self.assertRaisesRegex(
            ValueError,
            'invalid gpu request spec: 0.1',
            parse_resource,
            'cpu=2,memory=32Mi,disk=64Mi,gpu=0.1'
        )
        # When memory does not contain expected regex, raise an error
        self.assertRaisesRegex(
            ValueError,
            'invalid memory request spec: 32blah',
            parse_resource,
            'cpu=250m,memory=32blah,disk=64Mi,gpu=1'
        )
        # When resource name is unknown, raise an error
        self.assertRaisesRegex(
            ValueError,
            'unknown is not in the allowed list of resource types:',
            parse_resource,
            'cpu=250m,unknown=32Mi,disk=64Mi,gpu=1'
        )
        # When resource name is duplicated, raise an error
        self.assertRaisesRegex(
            ValueError,
            'The resource string contains duplicate resource names: cpu',
            parse_resource,
            'cpu=250m,cpu=32Mi,disk=64Mi,gpu=1'
        )
