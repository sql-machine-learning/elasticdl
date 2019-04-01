from concurrent import futures
import logging
import time

import grpc

from proto import master_pb2_grpc
from .servicer import MasterServicer

def main():
    logger = logging.getLogger("master")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=64))
    master_pb2_grpc.add_MasterServicer_to_server(
        MasterServicer(logger), server
    )
    server.add_insecure_port("[::]:50001")
    server.start()
    logger.warning("Server started")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.warning("Server stopping")
        server.stop(0)


if __name__ == "__main__":
    logging.basicConfig()
    main()
