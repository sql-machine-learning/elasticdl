import unittest

from elasticdl.python.common.k8s_volume import (
    parse,
    parse_volume_and_mount
)


class K8SVolumeTest(unittest.TestCase):
    def test_k8s_volume_parse(self):
        # parse works as expected on the allowed list of volume keys
        self.assertEqual(
            [{"claim_name": "c1", "mount_path": "/path1"}],
            parse("claim_name=c1,mount_path=/path1"),
        )

        # parse works as expected on the allowed list of volume dictionaries
        # with multiple volume configs
        self.assertEqual(
            [
                {"host_path": "c0", "mount_path": "/path0"},
                {"claim_name": "c1", "mount_path": "/path1"},
            ],
            parse(
                """host_path=c0,mount_path=/path0;
                claim_name=c1,mount_path=/path1"""
            ),
        )

        # parse works as expected with redundant semicolons
        self.assertEqual(
            [{"claim_name": "c1", "mount_path": "/path1"}],
            parse("claim_name=c1,mount_path=/path1;"),
        )

        # parse works as expected with redundant spaces
        self.assertEqual(
            [{"claim_name": "c1", "mount_path": "/path1"}],
            parse("  claim_name=c1,   mount_path = /path1 "),
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
            "claim_name=c1,mount_path=/path1,mount_path=/path2",
        )

    def test_parse_volume_and_mount(self):
        volume_config = """host_path=c0,mount_path=/path0;
        claim_name=c1,mount_path=/path1"""
        volumes, volume_mounts = parse_volume_and_mount(volume_config, "test")
        self.assertEqual(len(volumes), 2)
        self.assertEqual(volumes[0].host_path.path, "c0")
        self.assertEqual(volumes[1].persistent_volume_claim.claim_name, "c1")
        self.assertEqual(volume_mounts[0].mount_path, "/path0")
        self.assertEqual(volume_mounts[1].mount_path, "/path1")
