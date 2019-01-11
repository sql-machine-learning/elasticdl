"""
Swamp Coordinator GRPC server implementation.
"""
from concurrent import futures
import logging
import pickle
import random
import threading
import time

from google.protobuf.empty_pb2 import Empty
import grpc

from proto.service_pb2 import Model, PullResponse
import proto.service_pb2_grpc


class CoordinatorServicer(proto.service_pb2_grpc.CoordinatorServicer):
    """
    Server implementation
    """

    def __init__(self, model_selector):
        self._model_selector = model_selector

    def Push(self, request, context):
        if request.loss <= 0:
            raise ValueError("Invalid score: " + request.score)

        model = pickle.loads(request.model.torch_pickled)
        self._model_selector.add(request.trainer_id, model, request.loss)
        return Empty()

    def Pull(self, request, context):
        model, loss = self._model_selector.get(request.trainer_id)
        response = PullResponse(
            loss=loss, model=Model(torch_pickled=pickle.dumps(model))
        )
        return response


# TODO: call pytorch to eval model
def eval_torch_model(model):
    "Evaluate a torch model and return loss"
    return random.random()


class _ModelSelector(object):
    def __init__(self, *, max_pending, model_evaluator=eval_torch_model):
        self._exit = False
        self._max_pending = max_pending
        self._model_evaluator = model_evaluator
        # used as a lock to protect _best, and a cv for _pending_models.
        self._cv = threading.Condition()
        self._best = (None, float("inf"))
        # pending models are sorted by their loss value, in decresing order.
        self._pending_models = []

        self._eval_thread = threading.Thread(
            target=self._eval_model, name="eval_model_thread"
        )

    def start(self):
        "start model selector's eval thread"
        logging.info("starting model eval thread")
        self._eval_thread.start()

    def stop(self):
        "stop model selector's eval thread"
        logging.info("stopping model eval thread")
        with self._cv:
            self._exit = True
            self._cv.notify()
        self._eval_thread.join()

    def get(self, trainer_id):
        "get the current best model"
        with self._cv:
            model, loss = self._best
            logging.info(
                "trainer get model: id: %d loss: %f", trainer_id, loss
            )
            if self._best is None:
                logging.error("trainer get model error: id: %d", trainer_id)
            return model, loss

    # Fast admission check to reject bad models based only on loss value.
    # NOTE：We could consider multiple ways to relax the condition.
    def _admissible(self, loss, best_loss):
        return loss < best_loss

    def add(self, trainer_id, model, loss):
        "add new model to pending list to be evaluated"
        with self._cv:
            logging.info(
                "trainer add  model: id %d loss: %f", trainer_id, loss
            )
            _, best_loss = self._best
            if not self._admissible(loss, best_loss):
                logging.info(
                    "trainer model rejected by admission, loss %f "
                    "best_loss %f",
                    loss,
                    self._best[1],
                )
                return

            # Add model to pending list and sort it. We are not likely to have
            # many pending models so in place sort is plenty fast,
            self._pending_models.append((model, loss))
            self._pending_models.sort(key=lambda x: x[1], reverse=True)

            if len(self._pending_models) > self._max_pending:
                # pending queue full, drop the model with largest loss
                logging.info(
                    "model selector pending queue full, dropping model with "
                    "largest loss."
                )
                self._pending_models = self._pending_models[1:]
            self._cv.notify()

    def _eval_model(self):
        while True:
            model = None
            with self._cv:
                self._cv.wait_for(lambda: self._exit or self._pending_models)
                if self._exit:
                    return
                model = self._pending_models.pop()
            loss = self._model_evaluator(model)
            logging.info("evaluated model, loss: %f", loss)
            with self._cv:
                if loss < self._best[1]:
                    logging.info(
                        "updating best model: old loss: %f new loss: %f",
                        self._best[1],
                        loss,
                    )
                    # TODO: dump the best model.
                    self._best = (model, loss)
                # best loss changed, drop inadmissible models from
                # pending list.
                i = 0
                while i < len(self._pending_models) and not self._admissible(
                    self._pending_models[i][1], loss
                ):
                    i += 1
                logging.info(
                    "Dropping %d inadmissible models from pending list", i
                )
                self._pending_models = self._pending_models[i:]


def _serve():
    model_selector = _ModelSelector(max_pending=32)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    proto.service_pb2_grpc.add_CoordinatorServicer_to_server(
        CoordinatorServicer(model_selector), server
    )
    server.add_insecure_port("[::]:5000")

    model_selector.start()
    server.start()

    try:
        while True:
            time.sleep(24 * 60 * 60)
    except KeyboardInterrupt:
        server.stop(0)
        model_selector.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _serve()
