import argparse
import grpc

import tensorflow as tf

tf.enable_eager_execution()

from .worker import Worker


def _parse_args():
    parser = argparse.ArgumentParser(description="ElasticDL Worker")
    parser.add_argument("--master_addr", help="Master ip:port", required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    channel = grpc.insecure_channel(args.master_addr)

    # TODO: Init worker and do distributed train
    class _Model(object):
        def get_keras_model(self):
            return None

    worker = Worker(lambda: _Model(), lambda: None, None, channel=channel)

    while True:
        task = worker.get_task()
        print(task)
        if not task.shard_file_name:
            break
        worker.report_task_result(task.task_id, "")


if __name__ == "__main__":
    main()
