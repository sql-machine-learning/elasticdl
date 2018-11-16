from global_variables import *
from header import *
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

  def chunk_offset(self, chunk_num):
    """ Returns the offset of chunk in a file

    Arguments:
      chunk_num: The number of chunk in a file

    Returns:
      Chunk offset
    """
    return self._chunk_offsets[chunk_num]

  def chunk_len(self, chunk_num):
    """ Returns the total chunk data size
    
    Arguments:
      chunk_num: The number of chunk in a file  
    
    Returns:
      Total chunk size 
    """
    return self._chunk_lens[chunk_num]

  def chunk_records(self, chunk_num):
    """ Returns the record numbers in the chunk
  
    Arguments:
      chunk_num: The number of chunk in a file

    Returns:
      Total record number of the chunk
    """
    return self._chunk_records[chunk_num] 

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
