import unittest

from elasticdl.python.elasticdl.master.task_queue import (
    _EvaluationJob,
    _TaskQueue,
)
from elasticdl.proto import elasticdl_pb2


class TaskQueueTest(unittest.TestCase):
    def test_create_get(self):
        task_q = _TaskQueue({"f1": 10, "f2": 10}, {}, 3, 1)

        all_tasks = [
            ("f1", 0, 3, elasticdl_pb2.TRAINING, -1),
            ("f1", 3, 6, elasticdl_pb2.TRAINING, -1),
            ("f1", 6, 9, elasticdl_pb2.TRAINING, -1),
            ("f1", 9, 10, elasticdl_pb2.TRAINING, -1),
            ("f2", 0, 3, elasticdl_pb2.TRAINING, -1),
            ("f2", 3, 6, elasticdl_pb2.TRAINING, -1),
            ("f2", 6, 9, elasticdl_pb2.TRAINING, -1),
            ("f2", 9, 10, elasticdl_pb2.TRAINING, -1),
        ]

        # get all tasks out, each worker is assigned 2 tasks.
        got_tasks = [task_q.get(i // 2) for i in range(8)]

        # verify ids ranges from 1 to 8
        self.assertEqual(list(range(1, 9)), [k for k, _ in got_tasks])

        # verify tasks
        self.assertEqual(sorted([v._info() for _, v in got_tasks]), all_tasks)

        # no todo tasks, should return None
        self.assertEqual((-1, None), task_q.get(10))

        # report 6 task successes.
        for t in (1, 3, 5, 7, 2, 8):
            task_q.report(t, True)

        # there should be 2 doing tasks left.
        self.assertEqual(2, len(task_q._doing))

        # report a task failure
        task_q.report(list(task_q._doing.items())[0][0], False)
        self.assertEqual(1, len(task_q._doing))

        # recover tasks from a dead worker
        task_q.recover_tasks(list(task_q._doing.items())[0][1][0])
        self.assertEqual(0, len(task_q._doing))

        self.assertEqual(2, len(task_q._todo))

        id1, t1 = task_q.get(11)
        id2, t2 = task_q.get(12)
        task_q.report(id1, True)
        task_q.report(id2, True)

        self.assertTrue(task_q.finished())

    def test_epoch(self):
        task_q = _TaskQueue({"f1": 10, "f2": 10}, {}, 3, 2)

        epoch_tasks = [
            ("f1", 0, 3, elasticdl_pb2.TRAINING, -1),
            ("f1", 3, 6, elasticdl_pb2.TRAINING, -1),
            ("f1", 6, 9, elasticdl_pb2.TRAINING, -1),
            ("f1", 9, 10, elasticdl_pb2.TRAINING, -1),
            ("f2", 0, 3, elasticdl_pb2.TRAINING, -1),
            ("f2", 3, 6, elasticdl_pb2.TRAINING, -1),
            ("f2", 6, 9, elasticdl_pb2.TRAINING, -1),
            ("f2", 9, 10, elasticdl_pb2.TRAINING, -1),
        ]

        # get first epoch tasks
        got_tasks = [task_q.get(i // 2) for i in range(8)]
        self.assertEqual(
            sorted([v._info() for _, v in got_tasks]), epoch_tasks
        )

        # get second epoch tasks
        got_tasks = [task_q.get(i // 2) for i in range(8)]
        self.assertEqual(
            sorted([v._info() for _, v in got_tasks]), epoch_tasks
        )

    def test_evaluation_job(self):
        model_version = 1
        total_tasks = 5
        job = _EvaluationJob(model_version, total_tasks)
        self.assertEqual(0, job._completed_tasks)
        self.assertFalse(job.finished())

        # Now make 4 tasks finished
        for i in range(4):
            job.complete_task()
        self.assertEqual(4, job._completed_tasks)
        self.assertFalse(job.finished())

        # Job not finish yet, so not allow new evaluation job
        throttle_secs = 0
        job._start_time = 0
        latest_chkp_version = model_version + 1
        time_now_secs = 1
        self.assertFalse(
            job.ok_to_new_job(
                time_now_secs, throttle_secs, latest_chkp_version
            )
        )

        # One more task finishes
        job.complete_task()
        self.assertEqual(5, job._completed_tasks)
        self.assertTrue(job.finished())
        self.assertTrue(
            job.ok_to_new_job(
                time_now_secs, throttle_secs, latest_chkp_version
            )
        )

        # No new model checkpoint
        latest_chkp_version = job._model_version
        self.assertFalse(
            job.ok_to_new_job(
                time_now_secs, throttle_secs, latest_chkp_version
            )
        )
        latest_chkp_version = job._model_version + 1
        self.assertTrue(
            job.ok_to_new_job(
                time_now_secs, throttle_secs, latest_chkp_version
            )
        )

        # Need to wait for throttle secs
        throttle_secs = 2
        time_now_secs = job._start_time
        self.assertFalse(
            job.ok_to_new_job(
                time_now_secs, throttle_secs, latest_chkp_version
            )
        )
        time_now_secs = job._start_time + throttle_secs
        self.assertTrue(
            job.ok_to_new_job(
                time_now_secs, throttle_secs, latest_chkp_version
            )
        )


if __name__ == "__main__":
    unittest.main()
