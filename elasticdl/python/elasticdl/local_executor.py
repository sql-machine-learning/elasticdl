import os
import sys
from collections import namedtuple

import tensorflow as tf

from elasticdl.python.common.args import parse_envs
from elasticdl.python.common.constants import MetricsDictKey
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.common.model_utils import (
    get_dict_from_params_str,
    get_model_spec,
)
from elasticdl.python.data.reader.csv_reader import CSVDataReader
from elasticdl.python.data.reader.data_reader_factory import create_data_reader
from elasticdl.python.data.reader.odps_reader import ODPSDataReader
from elasticdl.python.data.reader.recordio_reader import RecordIODataReader
from elasticdl.python.master.evaluation_service import EvaluationJob

_MockedTask = namedtuple("Task", ["shard_name", "start", "end"])


class LocalExecutor:
    """Train the Keras model defined in ElasticDL model zoo locally
    using the custom training loop with Tensorflow native low-level API.
    """

    def __init__(self, args):
        envs = parse_envs(args.envs)
        self._init_environment(envs)

        (
            self.model_inst,
            self.dataset_fn,
            self.loss_fn,
            self.opt_fn,
            self.eval_metrics_fn,
            self.prediction_outputs_processor,
            self.custom_data_reader,
        ) = get_model_spec(
            model_zoo=args.model_zoo,
            model_def=args.model_def,
            dataset_fn=args.dataset_fn,
            loss=args.loss,
            optimizer=args.optimizer,
            eval_metrics_fn=args.eval_metrics_fn,
            model_params=args.model_params,
            prediction_outputs_processor="",
            custom_data_reader="",
        )
        self.opt = self.opt_fn()
        self.epoch = args.num_epochs
        self.evaluation_steps = args.evaluation_steps
        self.batch_size = args.minibatch_size
        self.data_reader_params = get_dict_from_params_str(
            args.data_reader_params
        )
        self.records_per_task = (
            args.minibatch_size * args.num_minibatches_per_task
        )
        self.data_reader = create_data_reader(
            data_origin=args.training_data,
            records_per_task=self.records_per_task,
            **self.data_reader_params
        )
        self.training_data = args.training_data
        self.validation_data = args.validation_data

    def _init_environment(self, envs):
        for key, value in envs.items():
            os.environ[key] = value

    def run(self):
        """Execute the training loop"""
        epoch = 0
        step = 0

        train_tasks = self._gen_tasks(self.training_data)
        validation_tasks = self._gen_tasks(self.validation_data)
        train_dataset = self._get_dataset(train_tasks)
        validation_dataset = self._get_dataset(validation_tasks)

        while epoch < self.epoch:
            for features, labels in train_dataset:
                loss = self._train(features, labels)
                logger.info("Loss = {}".format(loss))
                step += 1
                if (
                    self.evaluation_steps > 0
                    and step % self.evaluation_steps == 0
                ):
                    self._evaluate(validation_dataset)
            self._evaluate(validation_dataset)
            logger.info("Epoch {} end".format(epoch))
            epoch += 1

    def _train(self, features, labels):
        with tf.GradientTape() as tape:
            outputs = self.model_inst.call(features, training=True)
            loss = self.loss_fn(labels, outputs)
        grads = tape.gradient(loss, self.model_inst.trainable_variables)
        grads_and_vars = zip(grads, self.model_inst.trainable_variables)
        self.opt.apply_gradients(grads_and_vars, name=None)
        return loss

    def _evaluate(self, dataset):
        if dataset is None:
            logger.info("No validation dataset is configured")
            return
        eval_job = EvaluationJob(self.eval_metrics_fn(), -1)
        for features, labels in dataset:
            outputs = self.model_inst.call(features)
            if not isinstance(outputs, dict):
                outputs = {MetricsDictKey.MODEL_OUTPUT: outputs}
            eval_job.update_evaluation_metrics(outputs, labels)
        metrics = eval_job.get_evaluation_summary()
        logger.info("Evaluation metrics : {}".format(metrics))
        return metrics

    def _get_dataset(self, tasks):
        """
        Utilize tasks to creates a generator, which could be used to
        creating a `tf.data.Dataset` object in further.
        """

        def gen():
            for task in tasks:
                for data in self.data_reader.read_records(task):
                    if data:
                        yield data

        dataset = tf.data.Dataset.from_generator(
            gen, self.data_reader.records_output_types
        )
        dataset = self.dataset_fn(dataset, None, None)
        dataset = dataset.batch(self.batch_size)
        return dataset

    def _gen_tasks(self, data_dir):
        """Generate tasks to create a dataset.
        For CSVDataReader, a task will contains all records in a file.
        For RecordIODataReader, a task contains all records in a shard file.
        For ODPSDataReader, a task contains a portion of records in the table.
        """
        tasks = []
        if isinstance(self.data_reader, CSVDataReader):
            if os.path.exists(data_dir):
                # A task contains all records for a csv file
                tasks.append(_MockedTask(data_dir, 0, sys.maxsize))
        else:
            shards = self._create_shards(data_dir)
            if isinstance(self.data_reader, RecordIODataReader):
                for shard_name, (start_index, end_index) in shards.items():
                    tasks.append(
                        _MockedTask(shard_name, start_index, end_index)
                    )
            elif isinstance(self.data_reader, ODPSDataReader):
                for shard_name, (start_index, end_index) in shards.items():
                    tasks.append(
                        _MockedTask(
                            shard_name, start_index, end_index + start_index
                        )
                    )
        return tasks

    def _create_shards(self, data_origin):
        """Create shards
        Args:
            data_origin: A recordIO directory or a csv file path
            or an ODPS table name.
        """
        partition = self.data_reader_params.get("partition", None)
        return (
            create_data_reader(
                data_origin=data_origin,
                records_per_task=self.records_per_task,
                partition=partition,
            ).create_shards()
            if data_origin
            else {}
        )
