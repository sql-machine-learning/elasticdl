import os
import unittest

from elasticdl.python.common.constants import ODPSConfig
from elasticdl.python.common.odps_reader import ODPSReader


@unittest.skipIf(
    os.environ.get("ODPS_TESTS", "False") == "False",
    "ODPS environment is not configured",
)
class ODPSReaderTest(unittest.TestCase):
    def setUp(self):
        self._project = os.environ[ODPSConfig.PROJECT_NAME]
        self._access_id = os.environ[ODPSConfig.ACCESS_ID]
        self._access_key = os.environ[ODPSConfig.ACCESS_KEY]
        self._endpoint = os.environ[ODPSConfig.ENDPOINT]
        self._table = "chicago_taxi_train_data"

    def test_to_iterator(self):
        reader = ODPSReader(
            self._project,
            self._access_id,
            self._access_key,
            self._endpoint,
            self._table,
            None,
            4,
            None,
        )

        iterator = reader.to_iterator(1, 0, 200, 2, False, None)
        for batch in iterator:
            self.assertEqual(
                len(batch), 200, "incompatible size: %d" % len(batch)
            )


if __name__ == "__main__":
    unittest.main()
