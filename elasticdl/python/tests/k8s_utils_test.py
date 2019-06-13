import unittest

from elasticdl.python.elasticdl.common.k8s_utils import parse_resource


class K8SUtilsTest(unittest.TestCase):

    def test_parse_resource(self):
        self.assertEqual(
            {
                'cpu': '250m',
                'memory': '32Mi',
                'disk': '64Mi',
                'gpu': '128Mi',
            },
            parse_resource(
                'cpu=250m,memory=32Mi,disk=64Mi,gpu=128Mi'
            )
        )
        self.assertRaisesRegex(
            ValueError,
            'invalid cpu request spec:',
            parse_resource,
            'cpu=250blah,memory=32Mi,disk=64Mi,gpu=128Mi'
        )
        self.assertRaisesRegex(
            ValueError,
            'invalid memory request spec:',
            parse_resource,
            'cpu=250m,memory=32blah,disk=64Mi,gpu=128Mi'
        )
