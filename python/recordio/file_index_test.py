import unittest
from file_index import *
from writer import *
from reader import *
import os


class TestFileIndex(unittest.TestCase):
    """ Test file_index.py
    """

    def test_one_chunk(self):
        file_name = '/tmp/elasticflow/recordio/test_file'
        tmp_file = open(file_name, 'wb')
        writer = Writer(tmp_file, 1000, Compressor(1))
        writer.write('china')
        writer.write('usa')
        writer.write('russia')
        writer.flush()
        tmp_file.close()

        tmp_file = open(file_name, 'rb')
        index = FileIndex(tmp_file)
        tmp_file.close()
        os.remove(file_name)

        self.assertEqual(1, index.total_chunks())
        self.assertEqual(3, index.chunk_records(0))

    def test_two_chunk(self):
        file_name = '/tmp/elasticflow/recordio/test_file'
        tmp_file = open(file_name, 'wb')
        writer = Writer(tmp_file, 10, Compressor(1))
        writer.write('china')
        writer.write('usa')
        writer.write('russia')
        writer.flush()
        tmp_file.close()

        tmp_file = open(file_name, 'rb')
        index = FileIndex(tmp_file)
        tmp_file.close()
        os.remove(file_name)

        self.assertEqual(2, index.chunk_records(0))
        self.assertEqual(1, index.chunk_records(1))
        self.assertEqual(2, index.total_chunks())

    def test_usage(self):
        data_source = []
        data_source.append('china')
        data_source.append('usa')
        data_source.append('russia')
        data_source.append('india')
        data_source.append('thailand')
        data_source.append('finland')
        data_source.append('france')
        data_source.append('germany')
        data_source.append('poland')
        data_source.append('san marino')
        data_source.append('sweden')
        data_source.append('neuseeland')
        data_source.append('argentina')
        data_source.append('canada')
        data_source.append('ottawa')
        data_source.append('bogota')
        data_source.append('panama')
        data_source.append('united states')
        data_source.append('brazil')
        data_source.append('barbados')

        file_name = '/tmp/elasticflow/recordio/test_file'
        tmp_file = open(file_name, 'wb')
        writer = Writer(tmp_file, 20)

        for data in data_source:
            writer.write(data)
        writer.flush()
        tmp_file.close()

        parsed_data = []
        tmp_file = open(file_name, 'rb')
        index = FileIndex(tmp_file)

        for i in range(index.total_chunks()):
            reader = Reader(tmp_file, index.chunk_offset(i))
            while reader.has_next():
                parsed_data.append(reader.next())

        tmp_file.close()
        os.remove(file_name)

        self.assertEqual(20, len(parsed_data))

        for v1, v2 in zip(data_source, parsed_data):
            self.assertEqual(v1, v2)

    def test_readme_demo(self):
        data = open('demo.recordio', 'wb')
        max_chunk_size = 1024
        writer = Writer(data, max_chunk_size)
        writer.write('abc')
        writer.write('edf')
        writer.flush()
        data.close()

        data = open('demo.recordio', 'rb')
        index = FileIndex(data)
        self.assertEqual(2, index.total_records())
        data.close()
        os.remove('demo.recordio')


if __name__ == '__main__':
    unittest.main()
