import unittest
import tempfile
from data import TorchDataset 
from recordio import FileIndex 
from recordio import File


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

        temp = tempfile.NamedTemporaryFile()

        with File(temp.name, 'w') as rdio_w:
            for data in data_source:
                rdio_w.write(data)

        with TorchDataset(temp.name) as dataset:
            self.assertEqual(list(dataset), list(data_source))


if __name__ == '__main__':
    unittest.main()
