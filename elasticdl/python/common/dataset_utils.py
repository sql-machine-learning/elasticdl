import tensorflow as tf

from elasticdl.python.common.data_reader import RecordIODataReader


def create_dataset_from_tasks(tasks):
    """
    Returns a `tf.data.Dataset` from a list of `Task`s.
    """

    class _Generator:
        def __init__(self, tasks):
            self._tasks = tasks
            # TODO: Support any subclasses of `AbstractDataReader`
            self._data_reader = RecordIODataReader()

        def gen(self):
            for task in self._tasks:
                for data in self._data_reader.read_records(task):
                    if data:
                        yield data
                    else:
                        break

    generator = _Generator(tasks)
    dataset = tf.data.Dataset.from_generator(
        generator.gen, (tf.string), (tf.TensorShape([]))
    )
    return dataset
