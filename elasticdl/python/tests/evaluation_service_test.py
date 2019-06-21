import os
import tempfile
import unittest

import numpy as np

from elasticdl.python.elasticdl.common.ndarray import ndarray_to_tensor
from elasticdl.python.elasticdl.master.checkpoint_service import (
    Checkpoint,
    CheckpointService,
)
from elasticdl.python.elasticdl.master.evaluation_service import (
    _EvaluationJob,
    EvaluationService,
)
from elasticdl.python.elasticdl.master.task_queue import _TaskQueue


class EvaluationServiceTest(unittest.TestCase):
    def testEvaluationJob(self):
        model_version = 1
        total_tasks = 5
        latest_chkp_version = 2
        job = _EvaluationJob(model_version, total_tasks)
        self.assertEqual(0, job._completed_tasks)
        self.assertFalse(job.finished())
        self.assertFalse(job.ok_to_new_job(latest_chkp_version))

        # Now make 4 tasks finished
        for i in range(4):
            job.complete_task()
        self.assertEqual(4, job._completed_tasks)
        self.assertFalse(job.finished())
        self.assertFalse(job.ok_to_new_job(latest_chkp_version))

        # One more task finishes
        job.complete_task()
        self.assertEqual(5, job._completed_tasks)
        self.assertTrue(job.finished())
        self.assertTrue(job.ok_to_new_job(latest_chkp_version))

        # No new model checkpoint
        latest_chkp_version = job.model_version
        self.assertFalse(job.ok_to_new_job(latest_chkp_version))
        latest_chkp_version = job.model_version + 1
        self.assertTrue(job.ok_to_new_job(latest_chkp_version))

        # At the beginning, no metrics
        self.assertFalse(job._evaluation_metrics)

        # Start to report metrics
        evaluation_version = job.model_version + 1
        evaluation_metrics = {
            "mse": ndarray_to_tensor(np.array([100, 200], dtype=np.float32))
        }
        self.assertFalse(
            job.report_evaluation_metrics(
                evaluation_version, evaluation_metrics
            )
        )
        self.assertFalse(job._evaluation_metrics)
        evaluation_version = job.model_version
        self.assertTrue(
            job.report_evaluation_metrics(
                evaluation_version, evaluation_metrics
            )
        )
        # One more
        evaluation_metrics = {
            "mse": ndarray_to_tensor(np.array([300, 400], dtype=np.float32))
        }
        job.report_evaluation_metrics(evaluation_version, evaluation_metrics)
        self.assertTrue(
            np.array_equal(
                np.array([200, 300], dtype=np.float32),
                job.get_evaluation_summary().get("mse"),
            )
        )

    def testEvaluationService(self):
        with tempfile.TemporaryDirectory() as tempdir:
            chkp_dir = os.path.join(tempdir, "testEvaluationService")
            checkpoint_service = CheckpointService(chkp_dir, 5, 5)
            task_q = _TaskQueue({}, {"f1": 10, "f2": 10}, 3, 1)

            # Evaluation metrics will not be accepted if no evaluation ongoing
            evaluation_service = EvaluationService(
                checkpoint_service, None, task_q, 10, 20
            )
            evaluation_metrics = {
                "mse": ndarray_to_tensor(
                    np.array([100, 200], dtype=np.float32)
                )
            }
            self.assertFalse(
                evaluation_service.report_evaluation_metrics(
                    1, evaluation_metrics
                )
            )

            # No checkpoint available
            self.assertFalse(evaluation_service.try_to_create_new_round())

            # Add a checkpoint and we can start evaluation
            checkpoint_service._checkpoint_list.append(Checkpoint(1, "file1"))
            self.assertEqual(0, len(task_q._todo))
            self.assertTrue(evaluation_service.try_to_create_new_round())
            self.assertEqual(8, len(task_q._todo))
            self.assertFalse(evaluation_service._eval_job.finished())

            for i in range(8):
                self.assertFalse(evaluation_service._eval_job.finished())
                evaluation_service.complete_task()
            self.assertTrue(evaluation_service._eval_job.finished())
