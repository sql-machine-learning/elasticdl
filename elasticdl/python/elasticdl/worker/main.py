import argparse
import grpc
import logging

from elasticdl.python.elasticdl.worker.worker import Worker  # noqa
from elasticdl.python.elasticdl.common.constants import (
    GRPC_MAX_SEND_MESSAGE_LENGTH,
    GRPC_MAX_RECEIVE_MESSAGE_LENGTH
)


def _parse_args():
    parser = argparse.ArgumentParser(description="ElasticDL Worker")
    parser.add_argument(
        "--worker_id", help="Id unique to the worker", type=int, required=True
    )
    parser.add_argument("--master_addr", help="Master ip:port", required=True)
    parser.add_argument(
        "--model_file",
        help="Full file path of user defined neural model",
        required=True,
    )
    parser.add_argument(
        "--codec_file",
        default="elasticdl/python/data/codec/tf_example_codec.py",
        help="Codec file name",
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        default="WARNING",
        help="Set the logging level",
    )

    return parser.parse_args()


def main():
    args = _parse_args()
    channel = grpc.insecure_channel(
        args.master_addr,
        options=[
            ("grpc.max_send_message_length", GRPC_MAX_SEND_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", GRPC_MAX_RECEIVE_MESSAGE_LENGTH),
        ],
    )

    # Initialize logger
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)-8s "
        "[%(filename)s:%(lineno)d] %(message)s"
    )
    logging.getLogger().setLevel(args.log_level)

    worker = Worker(
        args.worker_id,
        args.model_file,
        channel=channel,
        codec_file=args.codec_file,
    )
    worker.run()


if __name__ == "__main__":
    main()
