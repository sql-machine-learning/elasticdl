from recordio.global_variables import *
from recordio.header import *
import os


class FileIndex(object):
    """ Parse and index the chunk meta info in a recordio file
    """

    def __init__(self, in_file):
        # File offset of each chunk
        self._chunk_offsets = []
        # Total size of each chunk (include chunk header)
        self._chunk_lens = []
        # Total record count of each chunk
        self._chunk_records = []
        # Total record count of a recordio file
        self._total_records = 0
        # Total chunk count of a recordio file
        self._total_chunks = 0
        # Input recordio file
        self._in_file = in_file

        # Do the parse process
        self._parse(in_file)

    def _parse(self, in_file):
        """ The actual parse and index process

        Arguments:
          in_file: A single recordio file
        """
        file_size = os.path.getsize(in_file.name)
        offset = 0
        header = Header()
        chunk_index = 0

        while offset < file_size:
            header.parse(in_file, offset)

            self._chunk_offsets.append(offset)
            self._chunk_lens.append(header.compress_size())
            self._chunk_records.append(header.total_count())
            self._total_records += header.total_count()

            offset += header.compress_size() + header_size
            chunk_index += 1
        self._total_chunks = chunk_index

    def locate_record(self, index):
        """ Locate the index of chunk and inner chunk for a record in recordio file.

        Arguments:
          index: The index of record in a recordio file cross chunk.
          (record index starts from zero to (total_records -1))

        Returns:
          (chunk_index, record_index): the index of chunk in recordio file and index of record in the chunk.
        """
        global_count = 0
        for chk_index in range(self._total_chunks):
            global_count += self._chunk_records[chk_index]
            if index < global_count:
                return chk_index, (
                    self._chunk_records[chk_index] - (global_count - index))
        return -1, -1

    def get_record(self, index):
        """ Returns the record by global index(cross chunk) in a recordio file.

        Arguments:
          index: The index of record in a recordio file cross chunk.

        Returns:
          record: A string representing the indexed record

        Raises:
          RuntimeError: index out of bounds
        """
        chunk_index, record_index = self.locate_record(index)
        if chunk_index == -1 or record_index == -1:
            raise RuntimeError(
                'record index out of bounds for index ' +
                str(index))

        chunk_offset = self.chunk_offset

    def chunk_offset(self, chunk_index):
        """ Returns the offset of chunk in a file

        Arguments:
          chunk_num: The number of chunk in a file

        Returns:
          Chunk offset
        """
        return self._chunk_offsets[chunk_index]

    def chunk_len(self, chunk_index):
        """ Returns the total chunk data size

        Arguments:
          chunk_num: The number of chunk in a file

        Returns:
          Total chunk size
        """
        return self._chunk_lens[chunk_index]

    def chunk_records(self, chunk_index):
        """ Returns the record numbers in the chunk

        Arguments:
          chunk_num: The number of chunk in a file

        Returns:
          Total record number of the chunk
        """
        return self._chunk_records[chunk_index]

    def total_records(self):
        """ Returns the total record number of the file

        Returns:
          Total record number of the file
        """
        return self._total_records

    def total_chunks(self):
        """ Returns the total number of chunks of the file

        Returns:
          Total number of chunks
        """
        return self._total_chunks
