import unittest

from elasticdl.python.elasticdl.common.k8s_utils import parse_resource


class K8SUtilsTest(unittest.TestCase):

    def test_parse_resource(self):
        # When cpu is non-numeric
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
        # When cpu is integer
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
        # When cpu is float
        self.assertEqual(
            {
                'cpu': '0.1',
                'memory': '32Mi',
                'disk': '64Mi',
                'gpu': '128Mi',
            },
            parse_resource(
                'cpu=0.1,memory=32Mi,disk=64Mi,gpu=1'
            )
        )
        # When cpu is non-numeric
        self.assertRaisesRegex(
            ValueError,
            'invalid processing units (cpu or gpu) request spec:',
            parse_resource,
            'cpu=250blah,memory=32Mi,disk=64Mi,gpu=1'
        )
        # When gpu is non-numeric
        self.assertRaisesRegex(
            ValueError,
            'invalid processing units (cpu or gpu) request spec:',
            parse_resource,
            'cpu=2,memory=32Mi,disk=64Mi,gpu=1blah'
        )
        # When memory does not contain expected regex
        self.assertRaisesRegex(
            ValueError,
            'invalid memory request spec:',
            parse_resource,
            'cpu=250m,memory=32blah,disk=64Mi,gpu=128Mi'
        )
