import argparse
import logging

import grpc

from elasticdl.python.common.constants import GRPC
from elasticdl.python.common.model_helper import get_model_file
from elasticdl.python.worker.worker import Worker


def _parse_args():
    parser = argparse.ArgumentParser(description="ElasticDL Worker")
    parser.add_argument(
        "--worker_id", help="Id unique to the worker", type=int, required=True
    )
    parser.add_argument("--master_addr", help="Master ip:port", required=True)
    parser.add_argument(
        "--model_def",
        help="The directory that contains user-defined model files "
        "or a specific model file",
        required=True,
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        default="INFO",
        help="Set the logging level",
    )

    return parser.parse_args()


def main():
    args = _parse_args()
    channel = grpc.insecure_channel(
        args.master_addr,
        options=[
            ("grpc.max_send_message_length", GRPC.MAX_SEND_MESSAGE_LENGTH),
            (
                "grpc.max_receive_message_length",
                GRPC.MAX_RECEIVE_MESSAGE_LENGTH,
            ),
        ],
    )

    # Initialize logger
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)-8s "
        "[%(filename)s:%(lineno)d] %(message)s"
    )
    logging.getLogger().setLevel(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting worker %d", args.worker_id)
    worker = Worker(
        args.worker_id, get_model_file(args.model_def), channel=channel
    )
    worker.run()


if __name__ == "__main__":
    main()
