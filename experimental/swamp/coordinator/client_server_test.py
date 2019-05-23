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
        self.eval_event.set()
        self.model_loss.clear()

    def tearDown(self):
        # signal the potential pending eval.
        self.eval_event.set()
        self.model_selector.reset()
        time.sleep(0.1)

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
        # Block eval
        self.eval_event.clear()
        model = self.makeModel(1, 0.1)
        self.client.push(model, 0.2)
        self.assertRaisesRegex(
            grpc.RpcError, "Model unavailable", self.client.pull
        )

    def testPullAfterPush(self):
        # server will eval the model and get a different loss.
        model = self.makeModel(1, 0.1)
        self.client.push(model, 0.2)
        time.sleep(0.1)

        pulled, loss = self.client.pull()
        self.assertAlmostEqual(loss, 0.1)
        self.assertEqual(pulled, model)

    def testPushIgnoreWorse(self):
        m1 = self.makeModel(1, 0.2)
        self.client.push(m1, 0.1)
        time.sleep(0.1)

        pulled, loss = self.client.pull()
        self.assertAlmostEqual(loss, 0.2)

        # This model will be ignored, though if it were re-evaled, it will be
        # a better one.
        m2 = self.makeModel(2, 0.1)

        self.client.push(m2, 0.3)
        time.sleep(0.1)

        pulled, loss = self.client.pull()
        self.assertAlmostEqual(loss, 0.2)
        self.assertEqual(pulled, m1)

    def testPushGetBetter(self):
        m1 = self.makeModel(1, 0.2)
        m2 = self.makeModel(2, 0.1)
        m3 = self.makeModel(3, 0.3)

        # m2 is the best after re-eval.
        self.client.push(m1, 0.01)
        self.client.push(m2, 0.01)
        self.client.push(m3, 0.01)
        time.sleep(0.1)

        pulled, loss = self.client.pull()
        self.assertAlmostEqual(loss, 0.1)
        self.assertEqual(pulled, m2)

    def testQueueingBehavior(self):
        # Block eval thread.
        self.eval_event.clear()

        m1 = self.makeModel(1, 0.4)
        m2 = self.makeModel(2, 0.05)
        m3 = self.makeModel(3, 0.1)
        m4 = self.makeModel(4, 0.2)
        m5 = self.makeModel(5, 0.3)

        # m1 will be pending in eval_model function
        self.client.push(m1, 0.4)
        # selector internal queue size = 3
        self.client.push(m2, 0.4)
        self.client.push(m3, 0.01)
        self.client.push(m4, 0.01)
        # this will evict m2, though were re-evaled, m2 is better.
        self.client.push(m5, 0.01)

        # Unlock eval thread.
        self.eval_event.set()
        time.sleep(0.1)

        pulled, loss = self.client.pull()
        self.assertAlmostEqual(loss, 0.1)
        self.assertEqual(pulled, m3)


if __name__ == "__main__":
    unittest.main()
