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
        with tempfile.NamedTemporaryFile() as tmp_file:
            writer = Writer(tmp_file, 1000, Compressor(1))
            writer.write('china')
            writer.write('usa')
            writer.write('russia')
            writer.flush()

            tmp_file.seek(0)
            index = FileIndex(tmp_file)

            self.assertEqual(1, index.total_chunks())
            self.assertEqual(3, index.chunk_records(0))

    def test_two_chunk(self):
        with tempfile.NamedTemporaryFile() as tmp_file:
            writer = Writer(tmp_file, 10, Compressor(1))
            writer.write('china')
            writer.write('usa')
            writer.write('russia')
            writer.flush()

            tmp_file.seek(0)
            index = FileIndex(tmp_file)

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

        with tempfile.NamedTemporaryFile() as tmp_file:
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

        self.assertEqual(data_source, parsed_data)

    def test_readme_demo(self):
        with tempfile.NamedTemporaryFile() as data:
            max_chunk_size = 1024
            writer = Writer(data, max_chunk_size)
            writer.write('abc')
            writer.write('edf')
            writer.flush()

            data.seek(0)
            index = FileIndex(data)
            self.assertEqual(2, index.total_records())

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

        with tempfile.NamedTemporaryFile() as tmp_file:
            writer = Writer(tmp_file, 20)

            for data in data_source:
                writer.write(data)
            writer.flush()

            tmp_file.seek(0)
            index = FileIndex(tmp_file)

            chunk_idx, record_idx = index.locate_record(0)
            self.assertEqual(chunk_idx, 0)
            self.assertEqual(record_idx, 0)

            chunk_idx, record_idx = index.locate_record(19)
            self.assertEqual(chunk_idx, 7)
            self.assertEqual(record_idx, 1)


if __name__ == '__main__':
    unittest.main()
