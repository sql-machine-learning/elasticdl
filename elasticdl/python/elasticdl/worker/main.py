import argparse
import grpc

import tensorflow as tf

tf.enable_eager_execution()

from elasticdl.worker.worker import Worker


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
        "--codec_type",
        default="bytes",
        choices=["tf_example", "bytes"],
        help="Type of codec(tf_example or bytes)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    channel = grpc.insecure_channel(args.master_addr)
    worker = Worker(
        args.worker_id,
        args.model_file,
        channel=channel,
        codec_type=args.codec_type,
    )

    worker.distributed_train()


if __name__ == "__main__":
    main()
