import unittest
import tempfile
from elasticdl.recordio import Header, Compressor


class TestHeader(unittest.TestCase):
    """ Test header.py
    """

    def test_write_and_parse(self):
        num_records = 1000
        checksum = 824863398
        compressor = Compressor.gzip
        compress_size = 10240

        with tempfile.NamedTemporaryFile() as tmp_file:
            header1 = Header(num_records, checksum, compressor, compress_size)
            header1.write(tmp_file)

            tmp_file.seek(0)
            header2 = Header()
            header2.parse(tmp_file, 0)

        self.assertEqual(num_records, header2.total_count())
        self.assertEqual(checksum, header2.checksum())
        self.assertEqual(compressor, header2.compressor())
        self.assertEqual(compress_size, header2.compress_size())


if __name__ == '__main__':
    unittest.main()
