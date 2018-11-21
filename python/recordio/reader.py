from recordio.global_variables import *
from recordio.chunk import *


class Reader(object):
    """ Read and parse the chunk of the given input file.
        a reader only response for processing one chunk in the recordio file.
    """

    def __init__(self, in_file, offset):
        # The chunk file.
        self._in_file = in_file

        # Parse the given chunk file.
        self._chunk = Chunk()
        # The start offset of the chunk in the recordio file
        self._chunk.parse(in_file, offset)

        # Total record count in the chunk
        self._total_count = self._chunk.total_count()
        # Current record index
        self._curr_index = 0

    def get(self, index):
        """ Return the record specified by the index
    
        Arguments:
          index: record index in the chunk
 
        Returns:
          String record value
 
        Raise:
          RuntimeError: index of out bounds
        """
        if index < 0 or index >= self._total_count: 
            raise RuntimeError('index out of bounds for index ' + str(index))

        return self._chunk.get(index)

    def next(self):
        """ Return the next chunk of the input file.

        Returns:
          The next string value in the chunk.

        Raise:
          RuntimeError: Reach the end of the chunk and no more any records
        """
        if self._curr_index < self._total_count:
            record = self._chunk.get(self._curr_index)
            self._curr_index += 1
            return record
        else:
            raise RuntimeError("no more any records in the chunk.")

    def has_next(self):
        """ Check if reach the end of the chunk.

        Returns:
          Bool value indicate whether there is next string value in the chunk.
        """
        return self._curr_index < self._total_count

    def total_count(self):
        """ Return the total record count.

        Returns:
          Total Record count in the current chunk.
        """
        return self._total_count
