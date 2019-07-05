import unittest

from elasticdl.python.common.k8s_volume import parse


class K8SVolumeTest(unittest.TestCase):
    def test_k8s_volume_parse(self):
        # parse works as expected on the allowed list of volume keys
        self.assertEqual(
            {
                "claim_name": "c1",
                "volume_name": "v1",
                "mount_path": "/path1",
            },
            parse(
                "claim_name=c1,volume_name=v1,mount_path=/path1"
            ),
        )
        # When volume key is unknown, raise an error
        self.assertRaisesRegex(
            ValueError,
            "unknown is not in the allowed list of volume keys:",
            parse,
            "claim_name=c1,unknown=v1,mount_path=/path1",
        )
        # When volume key is duplicated, raise an error
        self.assertRaisesRegex(
            ValueError,
            "The volume string contains duplicate volume key: mount_path",
            parse,
            "claim_name=c1,volume_name=v1,mount_path=/path1,mount_path=/path2",
        )
