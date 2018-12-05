from master import Master
import unittest
from unittest.mock import patch
from elasticdl.recordio.file_index import _ChunkData as C

class MockWorker(object):
    def __init__(self, q):
        self._q = q
    
    def run(self):
        
        pass

class MockRecordIoFile(object):
    def __init__(self, index):
        self._index = index
    def __enter__(self):
        return self
    def __exit__(self,exc_type, exc_val, exc_tb):
        return
    def get_index(self):
        return self._index



class MasterTest(unittest.TestCase):
    
    def setUp(self):
        recordio_data = {'f0': [C(0, 100, 2), C(200, 100, 3)], 'f1': [C(10, 200, 4), C(210, 200, 4)]}
        master =Master(recordio_data.keys(), num_epoch=3, max_trial=2)
        # patch Master's recordio calls to inject mock data
        with patch('master.File', autospec=True) as mock:
            mock.side_effect=[MockRecordIoFile(index) for index in recordio_data.values()]
            master.run()
    
    def test_all(self):
        pass

if __name__ == '__main__':
    unittest.main()
