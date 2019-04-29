import argparse
import grpc

import tensorflow as tf

tf.enable_eager_execution()

from elasticdl.worker.worker import Worker
from elasticdl.common.model_helper import load_user_model


def _parse_args():
    parser = argparse.ArgumentParser(description="ElasticDL Worker")
    parser.add_argument("--master_addr", help="Master ip:port", required=True)
    parser.add_argument(
        "--model-file",
        help="Full file path of user defined neural model",
        required=True,
    )
    parser.add_argument(
        "--model-class",
        help="The model class name defined in model file",
        required=True,
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    model_cls = load_user_model(args.model_file, args.model_class)
    channel = grpc.insecure_channel(args.master_addr)
    worker = Worker(model_cls, channel=channel)

    worker.distributed_train()


if __name__ == "__main__":
    main()
