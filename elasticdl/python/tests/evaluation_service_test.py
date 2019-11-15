import os
import tempfile
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Accuracy, MeanSquaredError

from elasticdl.python.common.constants import MetricsDictKey
from elasticdl.python.common.tensor import Tensor
from elasticdl.python.master.checkpoint_service import CheckpointService
from elasticdl.python.master.evaluation_service import (
    EvaluationService,
    _EvaluationJob,
)
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.master.task_dispatcher import _TaskDispatcher


def _eval_metrics_fn():
    return {
        "acc": Accuracy(),
        "mse": MeanSquaredError(),
        "acc_fn": lambda labels, outputs: tf.equal(
            tf.cast(outputs, tf.int32), tf.cast(labels, tf.int32)
        ),
    }


class EvaluationServiceTest(unittest.TestCase):
    @staticmethod
    def ok_to_new_job(job, latest_chkp_version):
        return job.finished() and latest_chkp_version > job.model_version

    def testEvaluationJob(self):
        model_version = 1
        total_tasks = 5
        latest_chkp_version = 2
        job = _EvaluationJob(_eval_metrics_fn(), model_version, total_tasks)
        self.assertEqual(0, job._completed_tasks)
        self.assertFalse(job.finished())
        self.assertFalse(self.ok_to_new_job(job, latest_chkp_version))

        # Now make 4 tasks finished
        for i in range(4):
            job.complete_task()
        self.assertEqual(4, job._completed_tasks)
        self.assertFalse(job.finished())
        self.assertFalse(self.ok_to_new_job(job, latest_chkp_version))

        # One more task finishes
        job.complete_task()
        self.assertEqual(5, job._completed_tasks)
        self.assertTrue(job.finished())
        self.assertTrue(self.ok_to_new_job(job, latest_chkp_version))

        # No new model checkpoint
        latest_chkp_version = job.model_version
        self.assertFalse(self.ok_to_new_job(job, latest_chkp_version))
        latest_chkp_version = job.model_version + 1
        self.assertTrue(self.ok_to_new_job(job, latest_chkp_version))

        # Start to report metrics
        evaluation_version = job.model_version + 1
        model_outputs = [
            Tensor(
                np.array([[1], [6], [3]], np.float32),
                name=MetricsDictKey.MODEL_OUTPUT,
            ).to_tensor_pb()
        ]
        labels = Tensor(np.array([[1], [0], [3]], np.float32)).to_tensor_pb()
        self.assertFalse(
            job.report_evaluation_metrics(
                evaluation_version, model_outputs, labels
            )
        )
        evaluation_version = job.model_version
        self.assertTrue(
            job.report_evaluation_metrics(
                evaluation_version, model_outputs, labels
            )
        )
        # One more
        self.assertTrue(
            job.report_evaluation_metrics(
                evaluation_version,
                [
                    Tensor(
                        np.array([[4], [5], [6], [7], [8]], np.float32),
                        name=MetricsDictKey.MODEL_OUTPUT,
                    ).to_tensor_pb()
                ],
                Tensor(
                    np.array([[7], [8], [9], [10], [11]], np.float32)
                ).to_tensor_pb(),
            )
        )
        expected_acc = 0.25
        evaluation_metrics = job.get_evaluation_summary()
        self.assertAlmostEqual(
            expected_acc, evaluation_metrics.get("acc").numpy()
        )
        self.assertAlmostEqual(
            expected_acc, evaluation_metrics.get("acc_fn").numpy()
        )
        self.assertAlmostEqual(10.125, evaluation_metrics.get("mse").numpy())

    def testEvaluationService(self):
        with tempfile.TemporaryDirectory() as tempdir:
            chkp_dir = os.path.join(tempdir, "testEvaluationService")
            checkpoint_service = CheckpointService(chkp_dir, 5, 5, True)
            task_d = _TaskDispatcher(
                {"f1": (0, 10), "f2": (0, 10)},
                {"f1": (0, 10), "f2": (0, 10)},
                {},
                3,
                1,
            )

            # Evaluation metrics will not be accepted if no evaluation ongoing
            evaluation_service = EvaluationService(
                None, task_d, 10, 20, 0, False, _eval_metrics_fn,
            )
            model_outputs = [
                Tensor(
                    np.array([1, 6, 3], np.float32),
                    name=MetricsDictKey.MODEL_OUTPUT,
                ).to_tensor_pb()
            ]
            labels = Tensor(np.array([1, 0, 3], np.float32)).to_tensor_pb()

            self.assertFalse(
                evaluation_service.report_evaluation_metrics(
                    1, model_outputs, labels
                )
            )

            # No checkpoint available
            self.assertFalse(evaluation_service.try_to_create_new_job())

            master = MasterServicer(
                2,
                2,
                None,
                task_d,
                init_var=[],
                checkpoint_filename_for_init="",
                checkpoint_service=checkpoint_service,
                evaluation_service=evaluation_service,
            )
            master.set_model_var("x", np.array([1.0, 1.0], dtype=np.float32))

            # Add an evaluation task and we can start evaluation
            self.assertEqual(8, len(task_d._todo))
            evaluation_service.add_evaluation_task(False)
            self.assertEqual(8, len(task_d._eval_todo))
            self.assertFalse(evaluation_service._eval_job.finished())

            for i in range(8):
                self.assertFalse(evaluation_service._eval_job.finished())
                evaluation_service.complete_task()
            self.assertTrue(evaluation_service._eval_job is None)
            self.assertFalse(evaluation_service.try_to_create_new_job())

    def testEvaluationOnly(self):
        task_d = _TaskDispatcher({}, {"f1": (0, 10), "f2": (0, 10)}, {}, 3, 1)

        evaluation_service = EvaluationService(
            None, task_d, 0, 0, 0, True, _eval_metrics_fn
        )
        task_d.set_evaluation_service(evaluation_service)

        master = MasterServicer(
            2,
            2,
            None,
            task_d,
            init_var=[],
            checkpoint_filename_for_init="",
            checkpoint_service=None,
            evaluation_service=evaluation_service,
        )
        master.set_model_var("x", np.array([1.0, 1.0], dtype=np.float32))

        self.assertEqual(8, len(task_d._eval_todo))
        for i in range(8):
            self.assertFalse(evaluation_service._eval_job.finished())
            evaluation_service.complete_task()
        self.assertTrue(evaluation_service._eval_job.finished())
