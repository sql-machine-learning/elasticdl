import os
import tempfile
import unittest
from collections import namedtuple

import numpy as np
from elasticdl.python.elasticdl.local_executor import LocalExecutor
from elasticdl.python.tests.test_utils import create_iris_csv_file

_MockedTask = namedtuple("Task", ["start", "end", "shard_name"])

_model_zoo_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../../model_zoo"
)

_iris_dnn_def = "odps_iris_dnn_model.odps_iris_dnn_model.custom_model"


class LocalExecutorArgs(object):
    def __init__(
        self,
        num_epochs,
        minibatch_size,
        training_data,
        validation_data,
        evaluation_steps,
        model_zoo,
        model_def,
        dataset_fn,
        loss,
        optimizer,
        eval_metrics_fn,
        model_params=None,
        prediction_outputs_processor=None,
        envs=None,
        data_reader_params=None,
        num_minibatches_per_task=None,
    ):
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.training_data = training_data
        self.validation_data = validation_data
        self.evaluation_steps = evaluation_steps
        self.model_zoo = model_zoo
        self.model_def = model_def
        self.dataset_fn = dataset_fn
        self.loss = loss
        self.optimizer = optimizer
        self.eval_metrics_fn = eval_metrics_fn
        self.model_params = model_params
        self.prediction_outputs_processor = prediction_outputs_processor
        self.envs = envs
        self.data_reader_params = data_reader_params
        self.num_minibatches_per_task = num_minibatches_per_task


class LocalExectorTest(unittest.TestCase):
    def test_train_model_local(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            num_records = 1000
            columns = [
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
                "class",
            ]
            training_data = create_iris_csv_file(
                size=num_records, columns=columns, temp_dir=temp_dir_name
            )
            validation_data = create_iris_csv_file(
                size=num_records, columns=columns, temp_dir=temp_dir_name
            )

            data_reader_params = (
                'columns=["sepal_length", "sepal_width", "petal_length",'
                '"petal_width", "class"]; seq=","'
            )
            args = LocalExecutorArgs(
                num_epochs=1,
                minibatch_size=32,
                training_data=training_data,
                validation_data=validation_data,
                evaluation_steps=10,
                model_zoo=_model_zoo_path,
                model_def=_iris_dnn_def,
                dataset_fn="dataset_fn",
                loss="loss",
                optimizer="optimizer",
                eval_metrics_fn="eval_metrics_fn",
                model_params="",
                prediction_outputs_processor="PredictionOutputsProcessor",
                data_reader_params=data_reader_params,
                num_minibatches_per_task=5,
            )
            local_executor = LocalExecutor(args)
            train_tasks = local_executor._gen_tasks(
                local_executor.training_data
            )
            validation_tasks = local_executor._gen_tasks(
                local_executor.validation_data
            )

            train_dataset = local_executor._get_dataset(train_tasks)
            for features, labels in train_dataset.take(1):
                loss = local_executor._train(features, labels)
                self.assertEqual(type(loss.numpy()), np.float32)

            validation_dataset = local_executor._get_dataset(validation_tasks)
            metrics = local_executor._evaluate(validation_dataset)
            self.assertEqual(list(metrics.keys()), ['accuracy'])

