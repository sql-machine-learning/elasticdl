from recordio.chunk import *


class Writer(object):
    """ Util class for user to transfer raw data into recordio chunk format.
    """

    def __init__(self, out_file, max_chunk_size, compressor=Compressor(3)):
        # Destination file for the chunk data.
        self._out_file = out_file
        # Max compressed data size per chunk
        self._max_chunk_size = max_chunk_size
        # Compression algorithm applied to chunk data
        self._compressor = compressor
        # Chunk instance used to store records
        self._chunk = Chunk()

    def write(self, record):
        """ Add a new record to the chunk.

        Arguments:
          record: bytes array representing a new record.

        Returns:
          True if write operation execute successfully.
        """
        if self._chunk.num_bytes() + len(record.encode(code_type)) > self._max_chunk_size:
            self._chunk.write(self._out_file, self._compressor)
            self._chunk.clear()

        self._chunk.add(record)

        return True

    def flush(self):
        """ flush the remaining records to chunk file.
        """
        self._chunk.write(self._out_file, self._compressor)
        self._chunk.clear()
