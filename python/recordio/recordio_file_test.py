import unittest
from recordio.recordio_file import *
import shutil
import os


class TestRecordIOFile(unittest.TestCase):
    """ Test recordio_file.py
    """

    def setUp(self):
        if not os.path.exists('/tmp/elasticdl'):
            os.mkdir('/tmp/elasticdl')
        if not os.path.exists('/tmp/elasticdl/recordio'):
            os.mkdir('/tmp/elasticdl/recordio')

    def tearDown(self):
        if os.path.exists('/tmp/elasticdl/recordio'):
            shutil.rmtree('/tmp/elasticdl/recordio')
   
    def test_read_by_index(self):
        rdio_w = RecordIOFile('/tmp/elasticdl/recordio/demo.recordio', 'w')
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

        for data in data_source:
            rdio_w.write(data)
        rdio_w.close()

        rdio_r = RecordIOFile('/tmp/elasticdl/recordio/demo.recordio', 'r')
        self.assertEqual(rdio_r.count(), 20)
        for index in range(len(data_source)):
            self.assertEqual(rdio_r.get(index), data_source[index])

        rdio_r.close() 

    def test_read_by_iter(self):
        rdio_w = RecordIOFile('/tmp/elasticdl/recordio/demo.recordio', 'w')
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

        for data in data_source:
            rdio_w.write(data)
        rdio_w.close()

        rdio_r = RecordIOFile('/tmp/elasticdl/recordio/demo.recordio', 'r')
        self.assertEqual(rdio_r.count(), 20)

        # use iter provided by recordio file
        iterator = rdio_r.iterator()
        index = 0
        while iterator.has_next():
            record = iterator.next()
            self.assertEqual(record, data_source[index])
            index += 1 

        rdio_r.close()


if __name__ == '__main__':
    unittest.main()
