from torch.utils.data import Dataset
import recordio


class TorchDataset(Dataset):
    """ Bridge the pytorch dataset and recordio file data source
    """

    def __init__(self, recordfile_path):
        # Create recordio file
        self._rdio = recordio.File(recordfile_path, 'r')

    def __enter__(self):
        """ For `with` statement
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ For `with` statement
        """
        self.close()

    def __iter__(self):
        """ For iterate operation
        Returns:
          Iterator of dataset
        """
        self._iter_idx = -1 
        return self
 
    def __next__(self):
        """ For iterate operation
        Returns:
          The next value in dataset

        Raise:
          StopIteration: Reach the end of dataset
        """
        self._iter_idx += 1  
        if self._iter_idx >= self._rdio.count():
            raise StopIteration

        return self.__getitem__(self._iter_idx)
         

    def __getitem__(self, index):
        """ Retrieve record data by index
        Arguments:
          index: record index in the recordio file

        Returns:
          Record value specified by index
        """
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
