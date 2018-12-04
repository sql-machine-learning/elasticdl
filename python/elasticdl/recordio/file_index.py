from elasticdl.recordio.global_variables import header_size
from elasticdl.recordio import Header
import os

# TODO: use @dataclass
class _ChunkData(object):
  def __init__(self, offset, len, num_record):
    # File offset of each chunk
    self.offset = offset
    # Total size of each chunk (include chunk header)
    self.len=len
    # Total record count of each chunk
    self.num_record = num_record

  def __str__(self):
    return 'offset: %s len: %s num_record: %s' % (self.offset, self.len, self.num_record)

    

class FileIndex(object):
    """ Parse and index the chunk meta info in a recordio file
    """

    def __init__(self, in_file):
        # Total record count of a recordio file
        self._total_records = 0
        # Input recordio file
        self._in_file = in_file
        self._chunk_data = []

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

        while offset < file_size:
            header.parse(in_file, offset)
            self._chunk_data.append(_ChunkData(offset, header.compress_size(), header.total_count()))
            self._total_records += header.total_count()
            offset += header.compress_size() + header_size

    def locate_record(self, index):
        """ Locate the index of chunk and inner chunk for a record in recordio file.

        Arguments:
          index: The index of record in a recordio file cross chunk.
          (record index starts from zero to (total_records -1))

        Returns:
          (chunk_index, record_index): the index of chunk in recordio file and index of record in the chunk.
        """
        global_count = 0
        for chk_index, chunk in enumerate(self._chunk_data):
            global_count += chunk.num_record
            if index < global_count:
                return chk_index, (chunk.num_record - (global_count - index))
        return -1, -1

    def chunk_offset(self, chunk_index):
        """ Returns the offset of chunk in a file

        Arguments:
          chunk_num: The number of chunk in a file

        Returns:
          Chunk offset
        """
        return self._chunk_data[chunk_index].offset

    def chunk_len(self, chunk_index):
        """ Returns the total chunk data size

        Arguments:
          chunk_num: The number of chunk in a file

        Returns:
          Total chunk size
        """
        return self._chunk_data[chunk_index].len

    def chunk_records(self, chunk_index):
        """ Returns the record numbers in the chunk

        Arguments:
          chunk_num: The number of chunk in a file

        Returns:
          Total record number of the chunk
        """
        return self._chunk_data[chunk_index].num_record

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
        return len(self._chunk_data)
    
    def __getitem__(self, chunk_index):
      return self._chunk_data[chunk_index]
    

