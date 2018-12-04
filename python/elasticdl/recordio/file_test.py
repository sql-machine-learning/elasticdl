import unittest
import tempfile
from elasticdl.recordio import File


class TestRecordIOFile(unittest.TestCase):
    """ Test file.py
    """

    def test_read_by_index(self):
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

        with File(tmp_file.name, 'w') as rdio_w:
            for data in data_source:
                rdio_w.write(data)

        with File(tmp_file.name, 'r') as rdio_r:
            self.assertEqual(list(rdio_r), list(data_source))

    def test_read_by_iter(self):
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

        with File(tmpfile_name, 'r') as rdio_r:
            self.assertEqual(list(rdio_r), list(data_source))


if __name__ == '__main__':
    unittest.main()
