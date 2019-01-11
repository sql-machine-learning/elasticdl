"""
Swamp Coordinator GRPC client implementation
"""
import pickle

import grpc

from proto.service_pb2 import Model, PushRequest, PullRequest
import proto.service_pb2_grpc


class Client(object):
    """
    Client implementation
    """
    def __init__(self, addr, trainer_id):
        channel = grpc.insecure_channel(addr)
        self._stub = proto.service_pb2_grpc.CoordinatorStub(channel)
        self._trainer_id = trainer_id

    def push(self, model, loss):
        "Push model and loss to server"
        self._stub.Push(
            PushRequest(
                model=Model(torch_pickled=pickle.dumps(model)),
                loss=loss,
                trainer_id=self._trainer_id,
            )
        )

    def pull(self):
        "Pull model and loss from server"
        response = self._stub.Pull(PullRequest(trainer_id=self._trainer_id))
        print("xxxxx", pickle.loads(response.model.torch_pickled))
        return pickle.loads(response.model.torch_pickled), response.loss
