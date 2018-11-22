import unittest
import tempfile
from recordio import FileIndex 
from recordio import Compressor
from recordio import Writer 
from recordio import Reader 


class TestFileIndex(unittest.TestCase):
    """ Test file_index.py
    """

    def test_one_chunk(self):
        tmp_file = tempfile.NamedTemporaryFile()
        writer = Writer(tmp_file, 1000, Compressor(1))
        writer.write('china')
        writer.write('usa')
        writer.write('russia')
        writer.flush()

        tmp_file.seek(0)
        index = FileIndex(tmp_file)
        tmp_file.close()

        self.assertEqual(1, index.total_chunks())
        self.assertEqual(3, index.chunk_records(0))

    def test_two_chunk(self):
        tmp_file = tempfile.NamedTemporaryFile()
        writer = Writer(tmp_file, 10, Compressor(1))
        writer.write('china')
        writer.write('usa')
        writer.write('russia')
        writer.flush()

        tmp_file.seek(0) 
        index = FileIndex(tmp_file)
        tmp_file.close()

        self.assertEqual(2, index.chunk_records(0))
        self.assertEqual(1, index.chunk_records(1))
        self.assertEqual(2, index.total_chunks())

    def test_usage(self):
        data_source = [
            'china',
            'usa',
            'russia',
            'india',
            'thailand',
            'finland',
            'france',
            'germany',
            'poland',
            'san marino',
            'sweden',
            'neuseeland',
            'argentina',
            'canada',
            'ottawa',
            'bogota',
            'panama',
            'united states',
            'brazil',
            'barbados']

        tmp_file = tempfile.NamedTemporaryFile()
        writer = Writer(tmp_file, 20)

        for data in data_source:
            writer.write(data)
        writer.flush()

        parsed_data = []
        tmp_file.seek(0)
        index = FileIndex(tmp_file)

        for i in range(index.total_chunks()):
            reader = Reader(tmp_file, index.chunk_offset(i))
            while reader.has_next():
                parsed_data.append(reader.next())

        tmp_file.close()

        self.assertEqual(20, len(parsed_data))

        for v1, v2 in zip(data_source, parsed_data):
            self.assertEqual(v1, v2)

    def test_readme_demo(self):
        data = tempfile.NamedTemporaryFile()
        max_chunk_size = 1024
        writer = Writer(data, max_chunk_size)
        writer.write('abc')
        writer.write('edf')
        writer.flush()

        data.seek(0)        
        index = FileIndex(data)
        self.assertEqual(2, index.total_records())
        data.close()

    def test_locate_record(self):
        data_source = [
            'china',
            'usa',
            'russia',
            'india',
            'thailand',
            'finland',
            'france',
            'germany',
            'poland',
            'san marino',
            'sweden',
            'neuseeland',
            'argentina',
            'canada',
            'ottawa',
            'bogota',
            'panama',
            'united states',
            'brazil',
            'barbados']

        tmp_file = tempfile.NamedTemporaryFile()
        writer = Writer(tmp_file, 20)

        for data in data_source:
            writer.write(data)
        writer.flush()

        tmp_file.seek(0)
        parsed_data = []
        index = FileIndex(tmp_file)
        tmp_file.close()


if __name__ == '__main__':
    unittest.main()
