import os
import tempfile
import unittest

import numpy as np

from elasticdl.python.common.ndarray import ndarray_to_tensor
from elasticdl.python.master.checkpoint_service import CheckpointService
from elasticdl.python.master.evaluation_service import (
    EvaluationService,
    _EvaluationJob,
)
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.master.task_dispatcher import _TaskDispatcher


class EvaluationServiceTest(unittest.TestCase):
    @staticmethod
    def ok_to_new_job(job, latest_chkp_version):
        return job.finished() and latest_chkp_version > job.model_version

    def testEvaluationJob(self):
        model_version = 1
        total_tasks = 5
        latest_chkp_version = 2
        job = _EvaluationJob(model_version, total_tasks)
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
                checkpoint_service, None, task_d, 10, 20, 0, False
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
            evaluation_service.add_evaluation_task(0)
            self.assertEqual(16, len(task_d._todo))
            self.assertFalse(evaluation_service._eval_job.finished())

            for i in range(8):
                self.assertFalse(evaluation_service._eval_job.finished())
                evaluation_service.complete_task()
            self.assertTrue(evaluation_service._eval_job is None)
            self.assertFalse(evaluation_service.try_to_create_new_job())

    def testEvaluationOnly(self):
        task_d = _TaskDispatcher({}, {"f1": (0, 10), "f2": (0, 10)}, {}, 3, 1)

        evaluation_service = EvaluationService(
            None, None, task_d, 0, 0, 0, True
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

        self.assertEqual(8, len(task_d._todo))
        for i in range(8):
            self.assertFalse(evaluation_service._eval_job.finished())
            evaluation_service.complete_task()
        self.assertTrue(evaluation_service._eval_job.finished())
