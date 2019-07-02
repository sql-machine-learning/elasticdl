import os
import unittest

from elasticdl.python.client.image_builder import (
    _create_dockerfile,
    _find_elasticdl_root,
    _generate_unique_image_name,
)


class DockerTest(unittest.TestCase):
    def test_find_elasticdl_root(self):
        rdm = os.path.join(_find_elasticdl_root(), "README.md")
        self.assertTrue(os.path.exists(rdm))
        with open(rdm, "r") as f:
            self.assertTrue(f.read().startswith("# ElasticDL:"))

    def test_generate_unique_image_name(self):
        self.assertTrue(
            _generate_unique_image_name(None).startswith("elasticdl:")
        )
        self.assertTrue(
            _generate_unique_image_name("").startswith("elasticdl:")
        )
        self.assertTrue(
            _generate_unique_image_name("proj").startswith("proj/elasticdl:")
        )
        self.assertTrue(
            _generate_unique_image_name("gcr.io/proj").startswith(
                "gcr.io/proj/elasticdl:"
            )
        )

    def test_create_dockerfile(self):
        self.assertTrue("COPY" in _create_dockerfile("/home/me/models"))
        self.assertTrue("COPY" in _create_dockerfile("file:///home/me/models"))
        self.assertTrue(
            "git clone" in _create_dockerfile("https://github.com/me/models")
        )
        with self.assertRaises(RuntimeError):
            _create_dockerfile("")
        with self.assertRaises(RuntimeError):
            _create_dockerfile(None)
