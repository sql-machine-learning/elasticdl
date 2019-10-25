import time
from concurrent import futures

import grpc

from elasticdl.proto import elasticdl_pb2_grpc
from elasticdl.python.common.args import parse_ps_args
from elasticdl.python.common.constants import GRPC
from elasticdl.python.common.log_utils import get_logger
from elasticdl.python.common.model_utils import (
    get_module_file_path,
    load_module,
)
from elasticdl.python.ps.parameters import Parameters
from elasticdl.python.ps.servicer import PserverServicer


def main():
    args = parse_ps_args()
    logger = get_logger("PS", level=args.log_level.upper())

    model_module = load_module(
        get_module_file_path(args.model_zoo, args.model_def)
    ).__dict__
    optimizer = model_module[args.optimizer]()

    logger.info("Starting PS pod with optimizer as %s", optimizer._name)

    # Create Parameters instance
    parameters = Parameters()

    # The pserver service
    logger.info("Starting pserver service")
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
        parameters,
        args.grads_to_wait,
        optimizer,
        lr_staleness_modulation=args.lr_staleness_modulation,
        use_async=args.use_async,
    )
    elasticdl_pb2_grpc.add_PserverServicer_to_server(pserver_servicer, server)
    server.add_insecure_port("[::]:{}".format(args.port))
    server.start()
    logger.info("RPC Server started at port: %d", args.port)

    try:
        while True:
            # TODO: add loop break condition
            time.sleep(30)
    except KeyboardInterrupt:
        logger.warning("Server stopping")

    server.stop(0)
    logger.info("RPC server stopped")
