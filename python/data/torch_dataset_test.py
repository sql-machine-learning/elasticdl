import unittest
import os
import shutil
from recordio.global_variables import *
from recordio.file_index import *
from recordio.recordio_file import RecordIOFile
from data.torch_dataset import *


class TestTorchDataset(unittest.TestCase):
    """ Test torch_dataset.py
    """

    def setUp(self):
        if not os.path.exists('/tmp/elasticdl'):
            os.mkdir('/tmp/elasticdl')
        if not os.path.exists('/tmp/elasticdl/recordio'):
            os.mkdir('/tmp/elasticdl/recordio')

    def tearDown(self):
        if os.path.exists('/tmp/elasticdl/recordio'):
            shutil.rmtree('/tmp/elasticdl/recordio')

    def test_dataset(self):
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

        dataset = TorchDataset('/tmp/elasticdl/recordio/demo.recordio')
        self.assertEqual(len(dataset), 20)
        for index in range(len(data_source)):
            self.assertEqual(dataset[index], data_source[index])
        dataset.close()


if __name__ == '__main__':
    unittest.main()
