import tensorflow as tf

from elasticdl.python.data.data_reader import create_data_reader


def create_dataset_from_tasks(tasks, data_reader_params=None):
    """
    Returns a `tf.data.Dataset` from a list of `Task`s.
    """

    class _Generator:
        def __init__(self, tasks):
            self._tasks = tasks
            if data_reader_params:
                self._data_reader = create_data_reader(data_origin=None, **data_reader_params)
            else:
                self._data_reader = create_data_reader(data_origin=None)

        def gen(self):
            for task in self._tasks:
                for data in self._data_reader.read_records(task):
                    if data:
                        yield data

    generator = _Generator(tasks)
    dataset = tf.data.Dataset.from_generator(
        generator.gen, generator._data_reader.records_output_types
    )
    return dataset
