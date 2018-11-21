from __future__ import absolute_import

from torch.utils.data import Dataset
from recordio.recordio_file import RecordIOFile

class TorchDataset(Dataset):
  """ Bridge the pytorch dataset and recordio file data source
  """

  def __init__(self, recordfile_path):
    # Create recordio file 
    self._rdio = RecordIOFile(recordfile_path, 'r')

  def __getitem__(self, index):
      """ Retrieve record data by index 
      Arguments:
        index: record index in the recordio file
  
      Returns:
        Record value specified by index

      Raise:
        RuntimeError: Index of of bounds
      """
      if index >= self._rdio.count():
          raise RuntimeError('Index out of bounds for index ' + str(index))

      return self._rdio.get(index)

  def __len__(self):
    """ Returns total record count in the recordio file

    Returns:
      total record count
    """
    return self._rdio.count() 

  def close(self):
    """ Close the recordio file
    """
    self._rdio.close()
