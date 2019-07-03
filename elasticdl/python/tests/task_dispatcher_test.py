import unittest

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.master.task_dispatcher import _TaskDispatcher


class TaskQueueTest(unittest.TestCase):
    def test_create_get(self):
        task_d = _TaskDispatcher({"f1": 10, "f2": 10}, {}, {}, 3, 1)

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
        got_tasks = [task_d.get(i // 2) for i in range(8)]

        # verify ids ranges from 1 to 8
        self.assertEqual(list(range(1, 9)), [k for k, _ in got_tasks])

        # verify tasks
        self.assertEqual(sorted([v._info() for _, v in got_tasks]), all_tasks)

        # no todo tasks, should return None
        self.assertEqual((-1, None), task_d.get(10))

        # report 6 task successes.
        for t in (1, 3, 5, 7, 2, 8):
            task_d.report(t, True)

        # there should be 2 doing tasks left.
        self.assertEqual(2, len(task_d._doing))

        # report a task failure
        task_d.report(list(task_d._doing.items())[0][0], False)
        self.assertEqual(1, len(task_d._doing))

        # recover tasks from a dead worker
        task_d.recover_tasks(list(task_d._doing.items())[0][1][0])
        self.assertEqual(0, len(task_d._doing))

        self.assertEqual(2, len(task_d._todo))

        id1, t1 = task_d.get(11)
        id2, t2 = task_d.get(12)
        task_d.report(id1, True)
        task_d.report(id2, True)

        self.assertTrue(task_d.finished())

    def test_epoch(self):
        task_d = _TaskDispatcher({"f1": 10, "f2": 10}, {}, {}, 3, 2)

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
        got_tasks = [task_d.get(i // 2) for i in range(8)]
        self.assertEqual(
            sorted([v._info() for _, v in got_tasks]), epoch_tasks
        )

        # get second epoch tasks
        got_tasks = [task_d.get(i // 2) for i in range(8)]
        self.assertEqual(
            sorted([v._info() for _, v in got_tasks]), epoch_tasks
        )


if __name__ == "__main__":
    unittest.main()
