import unittest
import os
from chunk import *


class TestHeader(unittest.TestCase):
    """ Test chunk.py
    """

    def test_add_and_get(self):
        chunk = Chunk()
        record1 = 'china'
        record2 = 'usa'
        record3 = 'russia'
        chunk.add(record1)
        chunk.add(record2)
        chunk.add(record3)

        self.assertEqual(chunk.get(0), record1)
        self.assertEqual(chunk.get(1), record2)
        self.assertEqual(chunk.get(2), record3)

    def test_clear(self):
        chunk = Chunk()
        record1 = 'china'
        record2 = 'usa'
        record3 = 'russia'
        chunk.add(record1)
        chunk.add(record2)
        chunk.add(record3)
        self.assertEqual(3, chunk.total_count())

        chunk.clear()
        self.assertEqual(0, chunk.total_count())

    def test_write_and_parse(self):
        chunk = Chunk()
        record1 = 'china'
        record2 = 'usa'
        record3 = 'russia'
        chunk.add(record1)
        chunk.add(record2)
        chunk.add(record3)

        file_name = '/tmp/elasticflow/recordio/test_file'
        tmp_file = open(file_name, 'wb')
        chunk.write(tmp_file, Compressor(2))
        tmp_file.close()

        tmp_file = open(file_name, 'rb')
        chunk.parse(tmp_file, 0)
        tmp_file.close()
        os.remove(file_name)

        self.assertEqual(chunk.get(0), record1)
        self.assertEqual(chunk.get(1), record2)
        self.assertEqual(chunk.get(2), record3)


if __name__ == '__main__':
    unittest.main()
