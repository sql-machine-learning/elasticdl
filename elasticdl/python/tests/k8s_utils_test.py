import unittest

from elasticdl.python.elasticdl.common.k8s_utils import (
    _parse_cpu,
    _parse_memory,
    parse_resource
)


class K8SUtilsTest(unittest.TestCase):
    def test_parse_cpu(self):
        self.assertAlmostEqual(0.001, _parse_cpu('1m'))
        self.assertAlmostEqual(0.01, _parse_cpu('10m'))
        self.assertAlmostEqual(0.1, _parse_cpu('100m'))
        self.assertAlmostEqual(1, _parse_cpu('1000m'))
        self.assertAlmostEqual(10, _parse_cpu('10000m'))
        self.assertAlmostEqual(0.001, _parse_cpu('0.001'))
        self.assertAlmostEqual(0.01, _parse_cpu('0.01'))
        self.assertAlmostEqual(0.1, _parse_cpu('0.1'))
        self.assertAlmostEqual(1, _parse_cpu('1'))
        self.assertAlmostEqual(10, _parse_cpu('10'))

    def test_parse_memory(self):
        self.assertEqual(0, _parse_memory('0'))
        self.assertEqual(1000, _parse_memory('1k'))
        self.assertEqual(1000 ** 2, _parse_memory('1M'))
        self.assertEqual(1000 ** 3, _parse_memory('1G'))
        self.assertEqual(1000 ** 4, _parse_memory('1T'))
        self.assertEqual(1000 ** 5, _parse_memory('1P'))
        self.assertEqual(1000 ** 6, _parse_memory('1E'))
        self.assertEqual(1024, _parse_memory('1Ki'))
        self.assertEqual(1024 ** 2, _parse_memory('1Mi'))
        self.assertEqual(1024 ** 3, _parse_memory('1Gi'))
        self.assertEqual(1024 ** 4, _parse_memory('1Ti'))
        self.assertEqual(1024 ** 5, _parse_memory('1Pi'))
        self.assertEqual(1024 ** 6, _parse_memory('1Ei'))
        self.assertEqual(15 * 1000, _parse_memory('15k'))
        self.assertEqual(20 * (1000 ** 2), _parse_memory('20M'))
        self.assertEqual(30 * (1000 ** 3), _parse_memory('30G'))
        self.assertEqual(40 * (1000 ** 4), _parse_memory('40T'))
        self.assertEqual(50 * (1000 ** 5), _parse_memory('50P'))
        self.assertEqual(60 * (1000 ** 6), _parse_memory('60E'))
        self.assertEqual(23 * 1024, _parse_memory('23Ki'))
        self.assertEqual(34 * (1024 ** 2), _parse_memory('34Mi'))
        self.assertEqual(45 * (1024 ** 3), _parse_memory('45Gi'))
        self.assertEqual(56 * (1024 ** 4), _parse_memory('56Ti'))
        self.assertEqual(67 * (1024 ** 5), _parse_memory('67Pi'))
        self.assertEqual(78 * (1024 ** 6), _parse_memory('78Ei'))

    def test_parse_resource(self):
        self.assertEqual(
            {
                'cpu': 1.0,
                'memory': 1024,
                'disk': 20480,
                'gpu': 1,
            },
            parse_resource(
                'cpu=1,memory=1024,disk=20480,gpu=1'
            )
        )
