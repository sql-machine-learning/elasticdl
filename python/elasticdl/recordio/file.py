from recordio import FileIndex
from recordio import Writer
from recordio import Reader


class File(object):
    """ Simple Wrapper for FileIndex, Writer and Reader for usability.
    """

    def __init__(self, file_path, mode, *, max_chunk_size=1024):
        """ Initialize according open mode

        Raises:
          ValueError: invalid open mode input param.
        """
        self._mode = mode
        if mode == 'r' or mode == 'read':
            self._data = open(file_path, 'rb')
            self._index = FileIndex(self._data)
        elif mode == 'w' or mode == 'write':
            self._data = open(file_path, 'wb')
            self._writer = Writer(self._data, max_chunk_size)
        else:
            raise ValueError('mode value should be \'read\' or \'write\'')

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

        Raises:
          RuntimeError: wrong open mode.
        """
        if self._mode != 'r' and self._mode != 'read':
            raise RuntimeError('Should be under read mode')

        # Starts from the first chunk
        self._chunk_index = 0
        # Initialize the first chunk
        self._reader = Reader(self._data, 0)

        return self

    def __next__(self):
        """ For iterate operation

        Returns:
          The next value in dataset

        Raises:
          StopIteration: Reach the end of dataset
        """
        if not self._reader.has_next() and (
                self._chunk_index + 1 >= self._index.total_chunks()):
            raise StopIteration

        # Switch to the next chunk
        if not self._reader.has_next():
            self._chunk_index += 1
            self._reader = Reader(self._data, self._chunk_index)

        return self._reader.next()

    def write(self, record):
        """ Write a record into recordio file.

        Arguments:
          record: Record value String.
         
        Raises:
          RuntimeError: wrong open mode.
        """
        if self._mode != 'w' and self._mode != 'write':
            raise RuntimeError('Should be under write mode')

        self._writer.write(record)

    def close(self):
        """ Close the data file
        """
        if self._mode == 'w' or self._mode == 'write':
            self._writer.flush()
        self._data.close()

    def get(self, index):
        """ Get the record string value specified by index

        Arguments:
          index: record index in the recordio file

        Returns:
          Record string value

        Raises:
          RuntimeError: wrong open mode.
        """
        if self._mode != 'r' and self._mode != 'read':
            raise RuntimeError('Should be under read mode')

        chunk_index, record_index = self._index.locate_record(index)
        chunk_offset = self._index.chunk_offset(chunk_index)
        reader = Reader(self._data, chunk_offset)
        return reader.get(record_index)

    def get_index(self):
        """ Returns the recordio file index

        Returns:
          Index of recordio file

        Raises:
          RuntimeError: wrong open mode.
        """
        if self._mode != 'r' and self._mode != 'read':
            raise RuntimeError('Should be under read mode')

        return self._index

    def count(self):
        """ Return total record count of the recordio file

        Returns:
          Total record count

        Raises:
          RuntimeError: wrong open mode.
        """
        if self._mode != 'r' and self._mode != 'read':
            raise RuntimeError('Should be under read mode')

        return self._index.total_records()
