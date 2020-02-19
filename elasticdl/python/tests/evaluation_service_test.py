import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Accuracy, MeanSquaredError

from elasticdl.python.common.constants import MetricsDictKey
from elasticdl.python.common.tensor_utils import ndarray_to_pb
from elasticdl.python.master.evaluation_service import (
    EvaluationJob,
    EvaluationService,
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
        job = EvaluationJob(_eval_metrics_fn(), model_version, total_tasks)
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

        model_outputs = {}
        model_outputs[MetricsDictKey.MODEL_OUTPUT] = ndarray_to_pb(
            np.array([[1], [6], [3]], np.float32)
        )
        labels = ndarray_to_pb(np.array([[1], [0], [3]], np.float32))
        job.report_evaluation_metrics(model_outputs, labels)
        job.report_evaluation_metrics(
            {
                MetricsDictKey.MODEL_OUTPUT: ndarray_to_pb(
                    np.array([[4], [5], [6], [7], [8]], np.float32)
                )
            },
            ndarray_to_pb(np.array([[7], [8], [9], [10], [11]], np.float32)),
        )
        expected_acc = 0.25
        evaluation_metrics = job.get_evaluation_summary()
        self.assertAlmostEqual(expected_acc, evaluation_metrics.get("acc"))
        self.assertAlmostEqual(expected_acc, evaluation_metrics.get("acc_fn"))
        self.assertAlmostEqual(10.125, evaluation_metrics.get("mse"))

    def testEvaluationService(self):
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

        _ = MasterServicer(2, task_d, evaluation_service=evaluation_service,)

        # No checkpoint available
        self.assertFalse(evaluation_service.try_to_create_new_job())

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

        _ = MasterServicer(2, task_d, evaluation_service=evaluation_service,)

        self.assertEqual(8, len(task_d._eval_todo))
        for i in range(8):
            self.assertFalse(evaluation_service._eval_job.finished())
            evaluation_service.complete_task()
        self.assertTrue(evaluation_service._eval_job.finished())

    def testNeedEvaluation(self):
        task_d = _TaskDispatcher(
            {"f1": (0, 10), "f2": (0, 10)},
            {"f1": (0, 10), "f2": (0, 10)},
            {},
            3,
            1,
        )

        evaluation_service = EvaluationService(
            None, task_d, 10, 0, 10, False, _eval_metrics_fn,
        )

        # Should add evaluation task and create eval job
        evaluation_service.add_evaluation_task_if_needed(
            master_locking=False, model_version=10
        )
        self.assertTrue(evaluation_service._eval_job is not None)
        self.assertEqual(evaluation_service._eval_checkpoint_versions, [])

        # Should ignore because version 10 is in the eval list
        evaluation_service.add_evaluation_task_if_needed(
            master_locking=False, model_version=10
        )
        self.assertEqual(evaluation_service._eval_checkpoint_versions, [])

        # Should append version 20 to the eval list
        evaluation_service.add_evaluation_task_if_needed(
            master_locking=False, model_version=20
        )
        self.assertEqual(evaluation_service._eval_checkpoint_versions, [20])

        # Should ignore version 10 because version 20 is already in eval list
        evaluation_service.add_evaluation_task_if_needed(
            master_locking=False, model_version=10
        )
        self.assertEqual(evaluation_service._eval_checkpoint_versions, [20])

        # Should append version 30 to the eval list
        evaluation_service.add_evaluation_task_if_needed(
            master_locking=False, model_version=30
        )
        self.assertEqual(
            evaluation_service._eval_checkpoint_versions, [20, 30]
        )

    def test_update_metric_by_small_chunks(self):
        labels = np.random.randint(0, 2, 1234)
        preds = np.random.random(1234)
        auc = tf.keras.metrics.AUC()
        auc.update_state(labels, preds)
        auc_value_0 = auc.result()

        auc.reset_states()
        EvaluationJob._update_metric_by_small_chunk(auc, labels, preds)
        auc_value_1 = auc.result()
        self.assertEquals(auc_value_0, auc_value_1)


if __name__ == "__main__":
    unittest.main()
