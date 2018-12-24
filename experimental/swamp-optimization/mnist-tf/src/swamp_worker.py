import numpy as np
import threading
import queue
import time
import os
import random
from random import shuffle
import tensorflow as tf
from tensorflow.python.ops import array_ops
from recordio.tensorflow_op import RecordIODataset
from recordio import File


# strip out ':0' part in name
def _extract_name(name):
    return name.split(":", 1)[0]


def get_files(file_dir):
    filenames = []
    for root, dirs, names in os.walk(file_dir, False):
        for filename in names:
            filenames.append(os.path.join(root, filename))
    return filenames


def get_work_list(filenames):
    work_list = []
    for i, f in enumerate(filenames):
        with File(f, "r") as fd:
            for chunk in fd.get_index():
                work_list.append((i, chunk.offset))
    shuffle(work_list)
    return work_list


class SwampWorker(object):
    def __init__(
        self,
        name=None,
        ps=None,
        umd=None,
        train_dir=None,
        test_dir=None,
        epoch=1,
        pull_model_probability=0.5,
        evaluation_frequency=4,
    ):
        assert ps
        assert umd
        self._name = name
        self._ps = ps
        self._umd = umd
        self._opt = umd.optimizer()
        self._model_initialized = False
        self._exiting = False
        self._graph = tf.Graph()
        self._runner = threading.Thread(target=self._run, name="worker_runner")
        self._epoch = epoch
        self._cur_epoch = 0
        self._cur_index = 0
        assert train_dir
        assert test_dir
        self._train_files = get_files(train_dir)
        self._train_work = get_work_list(self._train_files)
        self._test_files = get_files(test_dir)
        self._test_work = get_work_list(self._test_files)
        self._best_accuracy = 0
        self._best_batch_accuracy = 0
        self._pull_model_probability = pull_model_probability
        self._evaluation_frequency = evaluation_frequency

    # return filename, index
    def get_next_work(self):
        if self._cur_index == len(self._train_work):
            self._cur_index = 0
            self._cur_epoch += 1
        if self._cur_epoch >= self._epoch:
            raise queue.Empty
        work = self._train_work[self._cur_index]
        self._cur_index += 1
        return self._train_files[work[0]], work[1]

    def start(self):
        self._runner.start()

    def join(self):
        while self._exiting is not True:
            time.sleep(2)
        self._runner.join()

    def _run(self):
        worker_step = 0

        while not self._exiting:
            # get work from work queue
            try:
                data_file, file_offset = self.get_next_work()
            except queue.Empty:
                self._exiting = True
                break

            # TODO: how to config shuffle/batch parameter from user?
            shuffle_buffer_size = 1000
            batch_size = 64
            feed_dict = self._prepare_data_training(
                data_file, file_offset, batch_size, shuffle_buffer_size
            )

            # train loop for the dataset
            while True:
                try:
                    need_validate = (1 + worker_step) % self._evaluation_frequency == 0
                    self._train_step(feed_dict, need_validate, batch_size)
                    worker_step += 1
                except tf.errors.OutOfRangeError:
                    break

    def name(self):
        return self._name

    def _train_step(self, feed_dict, need_validate, batch_size):
        if need_validate:
            batch_accuracy, _ = self._sess.run(
                [self._accuracy, self._train_op], feed_dict=feed_dict
            )
        else:
            self._sess.run(self._train_op, feed_dict=feed_dict)
        push_test = False
        if need_validate:
            if batch_accuracy > self._best_batch_accuracy:
                self._best_batch_accuracy = batch_accuracy
                push_test = True
            else:
                # randomly pull model
                if random.random() < self._pull_model_probability:
                    self._pull_model()
                else:
                    push_test = True

            if push_test:
                # validation on test data
                accuracy = self._validate(batch_size)
                if self._best_accuracy < accuracy and self._ps.report_accuracy(
                    accuracy
                ):
                    self._push_model(accuracy)
                    self._best_accuracy = accuracy

    def _prepare_data_training(
        self, data_file, file_offset, batch_size, shuffle_buffer_size
    ):
        # create dataset from data_file, file_offset
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
                self._prepare_model_training(dataset)

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
                self._vars = trainable_vars
                self._var_placeholder = var_placeholder
                self._var_assign_op = var_assign_op
                self._sess = tf.Session(graph=self._graph)
                self._sess.run(tf.initializers.global_variables())

        # dataset initialization
        handle = self._sess.run(handle_op)
        self._sess.run(data_init_op)
        feed_dict = {self._iter_handle: handle}
        return feed_dict

    def _validate(self, batch_size):
        accum_acc = 0
        acc_num = 0
        for w in self._test_work:
            with self._graph.as_default():
                dataset = self._create_dataset(
                    self._test_files[w[0]], w[1], batch_size=batch_size
                )
                data_iter = dataset.make_initializable_iterator()
                data_init_op = data_iter.initializer
                handle_op = data_iter.string_handle()

            handle = self._sess.run(handle_op)
            self._sess.run(data_init_op)
            feed_dict = {self._iter_handle: handle}
            while True:
                try:
                    cur_acc = self._sess.run(self._accuracy, feed_dict=feed_dict)
                    accum_acc += cur_acc
                    acc_num += 1
                except tf.errors.OutOfRangeError:
                    break

        return accum_acc / acc_num

    def _push_model(self, accuracy):
        var_values = self._sess.run(self._vars)
        var_dict = {
            _extract_name(v.name): var_values[i] for i, v in enumerate(self._vars)
        }
        self._ps.push(accuracy, var_dict)

    def _pull_model(self):
        accuracy, var_values = self._ps.pull()

        assign_ops = [self._var_assign_op[v_name] for v_name in var_values]
        assign_feeds = {
            self._var_placeholder[v_name]: var_values[v_name] for v_name in var_values
        }
        self._sess.run(assign_ops, feed_dict=assign_feeds)
        self._best_accuracy = accuracy

    def _prepare_model_training(self, dataset):
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
        self._train_op = self._opt.apply_gradients(grads_and_vars)

        if hasattr(self._umd, "accuracy"):
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
        dataset = SwampWorker._create_recordio_dataset(data_file, file_offset)

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
