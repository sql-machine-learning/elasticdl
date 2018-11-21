from recordio.file_index import *
from recordio.writer import *
from recordio.reader import *


class RecordIOFile(object):
    """ Simple Wrapper for FileIndex, Writer and Reader for usability.
    """

    def __init__(self, file_path, mode, *, max_chunk_size=1024):
        """ Initialize according open mode
        """
        self._mode = mode
        if mode == 'r' or mode == 'read':
            self._data = open(file_path, 'rb')
            self._index = FileIndex(self._data)
        elif mode == 'w' or mode == 'write':
            self._data = open(file_path, 'wb')
            self._writer = Writer(self._data, max_chunk_size)
        else:
            raise RuntimeError('mode value should be \'read\' or \'write\'')

    def write(self, record):
        """
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

        Raise:
          RuntimeError: not under read mode
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

        Raise:
          RuntimeError: not under read mode
        """
        if self._mode != 'r' and self._mode != 'read':
            raise RuntimeError('Should be under read mode')

        return self._index

    def count(self):
        """ Return total record count of the recordio file

        Returns:
          Total record count

        Raise:
          RuntimeError: not under read mode
        """
        if self._mode != 'r' and self._mode != 'read':
            raise RuntimeError('Should be under read mode')

        return self._index.total_records()

    def iterator(self):
        """
        """
        if self._mode != 'r' and self._mode != 'read':
            raise RuntimeError('Should be under read mode')

        return Iterator(self._data, self._index)


class Iterator(object):
    """ Iterates over the recordio file
    """

    def __init__(self, data, index):
        # RecordIO data file
        self._data = data
        # Chunk index
        self._index = index
        # Starts from the first chunk
        self._chunk_index = 0
        # Initialize the first chunk
        self._reader = Reader(self._data, 0)

    def next(self):
        """ Return next record string value of the recordio file

        Returns:
          Next record value

        Raise:
          RuntimeError: Reach the end of the data file
        """
        # Switch to the next chunk
        if not self._reader.has_next():
            if self._chunk_index + 1 >= self._index.total_chunks():
                raise RuntimeError('Reach the end of file.')

            self._chunk_index += 1
            self._reader = Reader(self._data, self._chunk_index)

        return self._reader.next()

    def has_next(self):
        """ Check if there is any record

        Returns:
          True if not reach the end of file
        """
        return self._reader.has_next() or (
            self._chunk_index + 1 < self._index.total_chunks())
