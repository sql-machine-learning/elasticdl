from contextlib import closing

import recordio
import tensorflow as tf


def recordio_dataset(recordio_shards):
    """
    recordio_shards is a list of (recordio_file_name, start, end) tuples
    """

    class _Generator:
        def __init__(self, recordio_shards):
            self._shards = recordio_shards

        def gen(self):
            for s in self._shards:
                with closing(recordio.Scanner(s[0], s[1], s[2] - s[1])) as reader:
                    while True:
                        r = reader.record()
                        if r:
                            yield r
                        else:
                            break

    generator = _Generator(recordio_shards)
    ds = tf.data.Dataset.from_generator(
        generator.gen, (tf.string), (tf.TensorShape([]))
    )
    return ds
