import argparse
import grpc
import logging

import tensorflow as tf

tf.enable_eager_execution()

from elasticdl.python.elasticdl.worker.worker import Worker


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
    parser.add_argument(
        "--log_level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        type=str.upper,
        default='WARNING',
        help="Set the logging level",
    )
    parser.add_argument(
        "--task_type",
        default="training",
        choices=["training", "evaluation"],
        help="Type of worker task (training or evaluation). Default is training task.",
    )
    parser.add_argument(
        "--evaluate_steps",
        default=None,
        help="Evaluate the model by this many number of steps where the model is evaluated on one "
             "batch of samples for each step. By default, evaluation will continue until reaching the end of input.",
    )

    return parser.parse_args()


def main():
    args = _parse_args()
    channel = grpc.insecure_channel(args.master_addr)

    # Initialize logger
    logging.basicConfig(
        format='%(asctime)s %(name)s %(levelname)-8s '
        '[%(filename)s:%(lineno)d] %(message)s',
    )
    logging.getLogger().setLevel(args.log_level)

    worker = Worker(
        args.worker_id,
        args.model_file,
        channel=channel,
        codec_type=args.codec_type,
    )

    if args.task_type == "training":
        worker.distributed_train()
    else:
        worker.distributed_evaluate(steps=args.evaluate_steps)


if __name__ == "__main__":
    main()
