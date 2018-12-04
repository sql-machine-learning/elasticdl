import unittest
import tempfile
from elasticdl.data import TorchDataset 
from elasticdl.recordio import FileIndex 
from elasticdl.recordio import File


class TestTorchDataset(unittest.TestCase):
    """ Test torch_dataset.py
    """
   
    def test_dataset(self):
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

        # this tmp file will be closed in File.close()
        tmpfile_name = tempfile.NamedTemporaryFile().name
        with File(tmpfile_name, 'w') as rdio_w:
            for data in data_source:
                rdio_w.write(data)

        with TorchDataset(tmpfile_name) as dataset:
            self.assertEqual(list(dataset), list(data_source))


if __name__ == '__main__':
    unittest.main()
