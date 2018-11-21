import unittest
from recordio.writer import *
from recordio.reader import *
import os
import shutil


class TestHeader(unittest.TestCase):
    """ Test writer.py and reader.py
    """

    def setUp(self):
        if not os.path.exists('/tmp/elasticdl'):
            os.mkdir('/tmp/elasticdl')
        if not os.path.exists('/tmp/elasticdl/recordio'):
            os.mkdir('/tmp/elasticdl/recordio')

    def tearDown(self):
        if os.path.exists('/tmp/elasticdl/recordio'):
            shutil.rmtree('/tmp/elasticdl/recordio')

    def test_write_reader_no_flush(self):
        file_name = '/tmp/elasticdl/recordio/test_file'
        tmp_file = open(file_name, 'wb')
        writer = Writer(tmp_file, 1000, Compressor(1))
        writer.write('china')
        writer.write('usa')
        writer.write('russia')
        writer.flush()
        tmp_file.close()

        tmp_file = open(file_name, 'rb')
        reader = Reader(tmp_file, 0)
        self.assertEqual(3, reader.total_count())
        while reader.has_next():
            reader.next()
        tmp_file.close()
        os.remove(file_name)

    def test_write_reader_auto_flush(self):
        file_name = '/tmp/elasticdl/recordio/test_file'
        tmp_file = open(file_name, 'wb')
        writer = Writer(tmp_file, 10, Compressor(1))
        writer.write('china')
        writer.write('usa')
        writer.write('russia')
        writer.flush()
        tmp_file.close()

        tmp_file = open(file_name, 'rb')
        reader = Reader(tmp_file, 0)
        self.assertEqual(2, reader.total_count())
        while reader.has_next():
            reader.next()
        tmp_file.close()
        os.remove(file_name)


if __name__ == '__main__':
    unittest.main()
