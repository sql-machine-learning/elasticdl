import argparse
import grpc

import tensorflow as tf

tf.enable_eager_execution()

from .worker import Worker
from common.model_helper import load_user_model


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

    while True:
        task = worker.get_task()
        print(task)
        if not task.shard_file_name:
            break
        worker.report_task_result(task.task_id, "")


if __name__ == "__main__":
    main()
