from recordio.recordio.reader import RangeReader
import tensorflow as tf

class RangeReaderWrapper(object):

    def __init__(self, reader, f_desc):
        self._reader = reader
        self._f_desc = f_desc 

    def __next__(self):
        return tf.parse_single_example(self._reader.__next__(), self._f_desc)

    def __iter__(self):
        return self
