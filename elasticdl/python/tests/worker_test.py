import os
import tempfile
import unittest
from contextlib import closing

import numpy as np
import recordio
import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.constants import JobType
from elasticdl.python.common.model_helper import get_model_file, load_module
from elasticdl.python.master.checkpoint_service import CheckpointService
from elasticdl.python.master.evaluation_service import EvaluationService
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.master.task_dispatcher import _TaskDispatcher
from elasticdl.python.tests.in_process_master import InProcessMaster
from elasticdl.python.worker.worker import Worker

_model_file = get_model_file(
    os.path.dirname(os.path.realpath(__file__)), "test_module.custom_model"
)
m = load_module(_model_file).__dict__


def create_recordio_file(size):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with closing(recordio.Writer(temp_file.name)) as f:
        for _ in range(size):
            x = np.random.rand(1).astype(np.float32)
            y = 2 * x + 1
            example_dict = {
                "x": tf.train.Feature(float_list=tf.train.FloatList(value=x)),
                "y": tf.train.Feature(float_list=tf.train.FloatList(value=y)),
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=example_dict)
            )
            f.write(example.SerializeToString())
    return temp_file.name


class WorkerTest(unittest.TestCase):
    def distributed_train_and_evaluate(self, training=True):
        """
        Run distributed training and evaluation with a local master.
        grpc calls are mocked by local master call.
        """

        class _Master(InProcessMaster):
            def ReportGradient(self, req):
                if 2 < self._m._version < 80:
                    # For testing of retrain when gradient not accepted.
                    # Increase master version to reject the gradient.
                    self._m._version += 1
                return self._m.ReportGradient(req, None)

            def ReportEvaluationMetrics(self, req):
                if 2 < self._m._version < 80:
                    # Testing of evaluation retries. Increase the master
                    # version so the evaluation metrics will not be accepted.
                    self._m._version += 1
                return self._m.ReportEvaluationMetrics(req, None)

        job_type = (
            JobType.TRAINING_ONLY if training else JobType.EVALUATION_ONLY
        )
        batch_size = 16
        worker = Worker(
            1,
            job_type,
            batch_size,
            _model_file,
            model_def="test_module.custom_model",
            channel=None,
        )

        shards = {create_recordio_file(128): 128}
        if training:
            training_shards = shards
            evaluation_shards = {}
        else:
            training_shards = {}
            evaluation_shards = shards
        task_d = _TaskDispatcher(
            training_shards,
            evaluation_shards,
            {},
            records_per_task=64,
            num_epochs=1,
        )
        if not training:
            evaluation_service = EvaluationService(
                None, None, task_d, 0, 0, 0, True
            )
            task_d.set_evaluation_service(evaluation_service)
        else:
            evaluation_service = None
        master = MasterServicer(
            2,
            batch_size,
            worker._opt_fn(),
            task_d,
            init_var=[],
            checkpoint_filename_for_init="",
            checkpoint_service=None,
            evaluation_service=evaluation_service,
        )
        worker._stub = _Master(master)

        for var in worker._model.trainable_variables:
            master.set_model_var(var.name, var.numpy())

        worker.run()

        req = elasticdl_pb2.GetTaskRequest()
        req.worker_id = 1
        task = master.GetTask(req, None)
        # No more task.
        self.assertTrue(not task.shard_file_name)

    def test_distributed_train_tf_example(self):
        self.distributed_train_and_evaluate(training=True)

    def test_distributed_evaluate_tf_example(self):
        self.distributed_train_and_evaluate(training=False)

    def test_distributed_predict(self):
        init_var = m["custom_model"]().trainable_variables
        with tempfile.TemporaryDirectory() as tempdir:
            chkp_dir = os.path.join(tempdir, "testInitFromCheckpoint")
            os.makedirs(chkp_dir)
            master = MasterServicer(
                2,
                3,
                None,
                None,
                init_var=init_var,
                checkpoint_filename_for_init="",
                checkpoint_service=CheckpointService(chkp_dir, 2, 3, False),
                evaluation_service=None,
            )
            req = elasticdl_pb2.GetModelRequest()
            req.method = elasticdl_pb2.MINIMUM
            req.version = 0
            model = master.GetModel(req, None)
            master._checkpoint_service.save(master._version, model, False)

            chkp_file = master._checkpoint_service.get_checkpoint_path(
                master._version
            )
            prediction_shards = {create_recordio_file(128): 128}
            task_d = _TaskDispatcher(
                {}, {}, prediction_shards, records_per_task=64, num_epochs=1
            )

            # Create a MasterServicer whose model is initialized from a
            # checkpoint file for prediction tasks
            batch_size = 3
            master2 = MasterServicer(
                2,
                batch_size,
                None,
                task_d,
                init_var=init_var,
                checkpoint_filename_for_init=chkp_file,
                checkpoint_service=None,
                evaluation_service=None,
            )
            worker = Worker(
                1,
                JobType.PREDICT_ONLY,
                batch_size,
                _model_file,
                model_def="test_module.custom_model",
                channel=None,
            )
            worker._stub = InProcessMaster(master2)
            worker.run()

            req = elasticdl_pb2.GetTaskRequest()
            req.worker_id = 1
            task = master2.GetTask(req, None)
            # No more task.
            self.assertTrue(not task.shard_file_name)


if __name__ == "__main__":
    unittest.main()
