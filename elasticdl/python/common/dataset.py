import tensorflow as tf

from elasticdl.python.common.data_reader import RecordIODataReader


def recordio_dataset(recordio_shards):
    """
    Return a RecordIO dataset with records in recordio_shards.
    recordio_shards is a list of `Task` objects.
    """

    class _Generator:
        def __init__(self, recordio_shards):
            self._shards = recordio_shards
            self._recordio_data_reader = RecordIODataReader()

        def gen(self):
            for task in self._shards:
                return self._recordio_data_reader.read_records(task)

    generator = _Generator(recordio_shards)
    dataset = tf.data.Dataset.from_generator(
        generator.gen, (tf.string), (tf.TensorShape([]))
    )
    return dataset
