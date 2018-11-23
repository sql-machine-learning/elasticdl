import gzip
import os
from zlib import crc32
import snappy
from recordio import Header, Compressor
from recordio.global_variables import code_type, int_word_len, endian


class Chunk(object):
    """ A chunk is part of the original input file and is composed of one or more records
    """

    def __init__(self):
        # Current records stored in this chunk
        self._records = []
        # Total byte size of current records
        self._num_bytes = 0
        # Total count of the records
        self._total_count = 0

    def add(self, record):
        """ Add a new string record value to this chunk

        Arguments:
          record: A python3 byte class representing a record
        """
        byte_arr = record.encode(code_type)
        self._num_bytes += len(byte_arr)
        self._total_count += 1
        self._records.append(byte_arr)

    def get(self, index):
        """ Get a record string at the specified index position

        Arguments:
          index: The position of record in the records

        Returns:
          A string value represending the specified record

        Raises:
          RuntimeError: If the index is illegal
        """
        if index < 0 or index > len(self._records):
            raise IndexError(
                'illegal index value for the records size is {}'.format(len(self._records)))

        return self._records[index].decode(code_type)

    def clear(self):
        """ Clear and reset the current chunk for reuse
        """
        self._records = []
        self._num_bytes = 0
        self._total_count = 0

    def write(self, out_file, compressor):
        """ Write the chunk to the output file.

        Arguments:
          out_file: The output file of recordio format.

        Returns:
          True if the write operation execute successfully.
        """
        if self._total_count <= 0:
            return True

        # Compress the data according to compressor type
        uncompressed_bytes = bytearray()
        for record in self._records:
            rc_len = len(record)
            len_bytes = rc_len.to_bytes(int_word_len, endian)
            for lbyte in len_bytes:
                uncompressed_bytes.append(lbyte)
            for dbyte in record:
                uncompressed_bytes.append(dbyte)

        compressed_data = None

        # No compression
        if compressor is Compressor.no_compression:
            compressed_data = uncompressed_bytes
        # Snappy
        elif compressor is Compressor.snappy:
            compressed_data = snappy.compress(uncompressed_bytes)
        # Gzip
        elif compressor is Compressor.gzip:
            compressed_data = gzip.compress(uncompressed_bytes)
        # By default
        else:
            raise ValueError('invalid compressor')

        # Write chunk header into output file
        checksum = crc32(compressed_data)
        header = Header(
            self._total_count,
            checksum,
            compressor,
            len(compressed_data))
        header.write(out_file)

        # Write the compressed data body
        out_file.write(compressed_data)

        return True

    def parse(self, in_file, offset):
        """ Read and parse the next chunk from the input file.

        Arguments:
          in_file: The input file contains the original data.

        Returns:
          True if the parse operation execute successfully.

        Raises:
          RuntimeError: checksum check failed.
        """
        file_size = os.path.getsize(in_file.name)
        if offset < 0 or offset >= (file_size - int_word_len - 1):
            raise IndexError(
                'invalid offset {}, total file size {}'.format(
                    offset, file_size))

        in_file.seek(offset)

        header = Header()
        header.parse(in_file, offset)
        compressed_byte_arr = in_file.read(header.compress_size())
        uncompressed_byte_arr = None

        real_checksum = crc32(compressed_byte_arr)
        raw_checksum = header.checksum()

        if real_checksum != raw_checksum:
            raise RuntimeError(
                "checksum check failed for raw checksum {} and new checksum {}".format(
                    raw_checksum, real_checksum))

        compressor = header.compressor()
        # No compression
        if compressor is Compressor.no_compression:
            uncompressed_byte_arr = compressed_byte_arr
        # Snappy
        elif compressor is Compressor.snappy:
            uncompressed_byte_arr = snappy.uncompress(compressed_byte_arr)
        # Gzip
        elif compressor is Compressor.gzip:
            uncompressed_byte_arr = gzip.decompress(compressed_byte_arr)
        else:
            raise ValueError('invalid compressor')

        record_count = 0
        records = []
        curr_index = 0
        s_index = 0
        e_index = 0
        total_bytes = 0
        while record_count < header.total_count():
            rc_len = int.from_bytes(
                uncompressed_byte_arr[curr_index:curr_index + int_word_len], endian)
            s_index = curr_index + int_word_len
            e_index = s_index + rc_len
            total_bytes += rc_len

            # Read real data
            records.append(uncompressed_byte_arr[s_index:e_index])
            record_count += 1
            curr_index += int_word_len
            curr_index += rc_len

        self._num_bytes = total_bytes
        self._total_count = record_count
        self._records = records

        return True

    def total_count(self):
        """ Return current total records in the chunk

        Returns:
          current total records size
        """
        return self._total_count

    def num_bytes(self):
        """ Return current total bytes size in the chunk

        Returns:
          current total bytes size
        """
        return self._num_bytes
