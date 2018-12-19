import tensorflow as tf
import numpy as np
import threading
import queue
import time
from recordio.tensorflow_op import RecordIODataset
from tensorflow.python.ops import array_ops


# strip out ':0' part in name
def _extract_name(name):
    return name.split(":", 1)[0]


class Worker(object):
    def __init__(self, ps_client=None, work_queue=None, umd=None):
        assert ps_client
        assert work_queue
        assert umd
        self._ps_client = ps_client
        self._work_queue = work_queue
        self._umd = umd
        self._opt = umd.optimizer()
        self._model_initialized = False
        self._exiting = False
        self._graph = tf.Graph()
        self._runner = threading.Thread(target=self._run, name="worker_runner")

    def start(self):
        self._runner.start()

    def join(self):
        self._exiting = True
        self._runner.join()

    def _run(self):
        base_step = -1
        sub_step = 0

        while not self._exiting:
            # get work from work queue
            try:
                work_id, data_file, file_offset = self._work_queue.get_work(timeout=2.0)
            except queue.Empty:
                continue

            # create dataset from data_file, file_offset
            # TODO: how to config shuffle/batch parameter from user?
            shuffle_buffer_size = 1000
            batch_size = 16
            with self._graph.as_default():
                dataset = self._create_dataset(
                    data_file,
                    file_offset,
                    shuffle_buffer_size=shuffle_buffer_size,
                    batch_size=batch_size,
                )

                data_iter = dataset.make_initializable_iterator()
                data_init_op = data_iter.initializer
                handle_op = data_iter.string_handle()

                # init model if needed
                if not self._model_initialized:
                    self._prepare_for_training(dataset)

                    # create name,variable dict
                    # strip out the ':0' part in name
                    trainable_vars = tf.trainable_variables()
                    var_dict = {_extract_name(v.name): v for v in trainable_vars}
                    var_placeholder = {
                        _extract_name(v.name): array_ops.placeholder(dtype=v.dtype)
                        for v in trainable_vars
                    }
                    var_assign_op = {
                        name: tf.assign(var_dict[name], var_placeholder[name])
                        for name in var_dict
                    }
                    sess = tf.Session(graph=self._graph)
                    sess.run(tf.initializers.global_variables())

            # dataset initialization
            handle = sess.run(handle_op)
            sess.run(data_init_op)
            feed_dict = {self._iter_handle: handle}

            # train loop for the dataset
            # TODO: add pull/push frequency. pull/push for every iteration for now.
            while True:
                try:
                    # pull and update variable values
                    new_base_step, var_values = self._ps_client.pull(min_step=base_step + 1) 
                        
                    assign_ops = [var_assign_op[v_name] for v_name in var_values]
                    assign_feeds = {
                        var_placeholder[v_name]: var_values[v_name]
                        for v_name in var_values
                    }
                    sess.run(assign_ops, feed_dict=assign_feeds)

                    if self._accuracy is not None and base_step % 100 == 0:
                        grads, accuracy = sess.run([self._grads, self._accuracy], feed_dict=feed_dict)
                        print('Accuracy [%d]: %1.6g' % (base_step, accuracy))
                    else:
                        # compute grads
                        grads = sess.run(self._grads, feed_dict=feed_dict)

                    # push
                    self._ps_client.push(sub_step=sub_step, grads=grads)
                    base_step = new_base_step
                except tf.errors.OutOfRangeError:
                    break

            # report to master work done
            self._work_queue.work_done(work_id, True)

    def _prepare_for_training(self, dataset):
        # create placeholder for the dataset
        self._iter_handle = tf.placeholder(tf.string, shape=[], name="iter_handler")
        data_iter = tf.data.Iterator.from_string_handle(
            self._iter_handle,
            dataset.output_types,
            dataset.output_shapes,
            output_classes=dataset.output_classes,
        )
        next_data = data_iter.get_next()

        # TODO: need to provide a method to connect dataset and the model input.
        # here assume dataset has two items: [0] for model input, [1] for label.
        self._forward_result = self._umd.forward(next_data[0])
        self._loss = self._umd.loss(self._forward_result, next_data[1])
        grads_and_vars = self._opt.compute_gradients(self._loss)
        self._grads = {_extract_name(gv[1].name): gv[0] for gv in grads_and_vars}

        if hasattr(self._umd, 'accuracy'):
            self._accuracy = self._umd.accuracy(self._forward_result, next_data[1])
        else:
            self._accuracy = None

        self._model_initialized = True

    @staticmethod
    def _create_recordio_dataset(data_file, file_offset):
        dataset = RecordIODataset(data_file, file_offset)
        return dataset

    def _create_dataset(
        self, data_file, file_offset, shuffle_buffer_size=0, batch_size=1
    ):
        dataset = Worker._create_recordio_dataset(data_file, file_offset)

        # map with umd.raw_data_transform_by_py
        dataset = dataset.map(
            lambda data: tuple(
                tf.py_func(
                    self._umd.raw_data_transform_by_py,
                    [data],
                    self._umd.transformed_data_types(),
                )
            )
        )

        # map with umd.data_process_tf_func if exists
        if hasattr(self._umd, "data_preprocess_by_tf"):
            dataset = dataset.map(self._umd.data_preprocess_by_tf)

        # shuffle and batch if needed
        if shuffle_buffer_size:
            dataset = dataset.shuffle(shuffle_buffer_size)
        if batch_size > 1:
            dataset = dataset.batch(batch_size)

        return dataset
