import time
from concurrent import futures

import grpc

from elasticdl.proto import elasticdl_pb2_grpc
from elasticdl.python.common.constants import GRPC
from elasticdl.python.common.grpc_utils import build_channel
from elasticdl.python.common.log_utils import get_logger
from elasticdl.python.common.model_utils import (
    get_module_file_path,
    load_module,
)
from elasticdl.python.ps.parameters import Parameters
from elasticdl.python.ps.servicer import PserverServicer


class ParameterServer(object):
    def __init__(self, args):
        self.logger = get_logger("PS", level=args.log_level.upper())

        self.grads_to_wait = args.grads_to_wait
        self.lr_staleness_modulation = args.lr_staleness_modulation
        self.use_async = args.use_async
        self.port = args.port
        model_module = load_module(
            get_module_file_path(args.model_zoo, args.model_def)
        ).__dict__
        self.optimizer = model_module[args.optimizer]()
        # Create Parameters instance
        self.parameters = Parameters()
        if args.master_addr is None:
            raise ValueError("master_addr is missing for parameter servers")
        self.master_channel = build_channel(args.master_addr)
        self.evaluation_steps = args.evaluation_steps

    def prepare(self):
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=64),
            options=[
                ("grpc.max_send_message_length", GRPC.MAX_SEND_MESSAGE_LENGTH),
                (
                    "grpc.max_receive_message_length",
                    GRPC.MAX_RECEIVE_MESSAGE_LENGTH,
                ),
            ],
        )
        pserver_servicer = PserverServicer(
            self.parameters,
            self.grads_to_wait,
            self.optimizer,
            lr_staleness_modulation=self.lr_staleness_modulation,
            use_async=self.use_async,
            evaluation_steps=self.evaluation_steps,
            master_channel=self.master_channel,
        )
        elasticdl_pb2_grpc.add_PserverServicer_to_server(
            pserver_servicer, server
        )
        server.add_insecure_port("[::]:{}".format(self.port))
        server.start()
        self.server = server
        self.logger.info("RPC Server started at port: %d", self.port)

    def run(self):
        try:
            while True:
                # TODO: add loop break condition
                time.sleep(30)
        except KeyboardInterrupt:
            self.logger.warning("Server stopping")

        self.server.stop(0)
        self.logger.info("RPC server stopped")
