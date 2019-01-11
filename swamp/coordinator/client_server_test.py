import threading
import time
import unittest

import grpc

from client import Client
from server import _serve, _ModelSelector

# Used for mocking model evaluation.


class ClientServerTest(unittest.TestCase):
    # Mock model evaluation.
    @classmethod
    def eval_model(cls, m):
        print("eval model:", m["id"])
        cls.eval_event.wait()
        model, loss = cls.model_loss[m["id"]]
        assert model == m
        cls.eval_event.clear()
        return loss

    @classmethod
    def setUpClass(cls):
        # Used to synchronize eval thread and test thread
        cls.eval_event = threading.Event()
        # Used to mock model loss
        cls.model_loss = {}

        # Used to signal sever to exit.
        cls.event = threading.Event()
        cls.model_selector = _ModelSelector(
            max_pending=3, model_evaluator=cls.eval_model
        )
        # Run server in a thread
        threading.Thread(
            target=_serve, args=[5000, cls.model_selector, cls.event]
        ).start()
        # wait for server to start
        time.sleep(1)
        # Create a client to be shared among tests.
        cls.client = Client("localhost:5000", 1234)

    @classmethod
    def tearDownClass(cls):
        cls.event.set()

    def setUp(self):
        # For convenience
        self.client = self.__class__.client
        self.eval_event = self.__class__.eval_event
        self.model_loss = self.__class__.model_loss
        self.get_pending = (
            self.__class__.model_selector._get_num_pending_for_test
        )
        self._model_id = 0
        self.eval_event.clear()
        self.model_loss.clear()

    def tearDown(self):
        # TODO: theoratically we should a semaphore instead of an event here,
        # but realistally a deadlock here should not occur.

        self.model_selector.reset()
        # signal the potential pending eval.
        self.eval_event.set()

    def makeModel(self, x, loss):
        self._model_id += 1
        model = {"id": self._model_id, "x": x}
        self.model_loss[model["id"]] = (model, loss)
        return model

    def testPullBeforePush(self):
        self.assertRaisesRegex(
            grpc.RpcError, "Model unavailable", self.client.pull
        )

    def testBadLoss(self):
        model = self.makeModel(1, 0.1)
        self.assertRaisesRegex(
            grpc.RpcError, "Invalid loss", self.client.push, model, -0.1
        )

    def testPullBeforeFirstEval(self):
        model = self.makeModel(1, 0.1)
        self.client.push(model, 0.2)
        self.assertRaisesRegex(
            grpc.RpcError, "Model unavailable", self.client.pull
        )

    def testPullAfterPush(self):
        # server will eval the model and get a different loss.
        model = self.makeModel(1, 0.1)
        print(model)
        # unblock eval
        self.eval_event.set()
        self.client.push(model, 0.2)
        time.sleep(0.1)

        pulled, loss = self.client.pull()
        print(pulled)
        print(loss)
        self.assertAlmostEqual(loss, 0.1)
        self.assertEqual(pulled, model)


if __name__ == "__main__":
    unittest.main()
