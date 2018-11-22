import unittest
import tempfile
from recordio import Compressor
from recordio import Writer 
from recordio import Reader

class TestHeader(unittest.TestCase):
    """ Test writer.py and reader.py
    """

    def test_write_reader_no_flush(self):
        tmp_file = tempfile.NamedTemporaryFile()
        writer = Writer(tmp_file, 1000, Compressor(1))
        writer.write('china')
        writer.write('usa')
        writer.write('russia')
        writer.flush()

        tmp_file.seek(0)
        reader = Reader(tmp_file, 0)
        self.assertEqual(3, reader.total_count())
        while reader.has_next():
            reader.next()
        tmp_file.close()

    def test_write_reader_auto_flush(self):
        tmp_file = tempfile.NamedTemporaryFile()
        writer = Writer(tmp_file, 10, Compressor(1))
        writer.write('china')
        writer.write('usa')
        writer.write('russia')
        writer.flush()

        tmp_file.seek(0)
        reader = Reader(tmp_file, 0)
        self.assertEqual(2, reader.total_count())
        while reader.has_next():
            reader.next()
        tmp_file.close()


if __name__ == '__main__':
    unittest.main()
