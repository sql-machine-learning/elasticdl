import argparse
import grpc
import logging

import tensorflow as tf

tf.enable_eager_execution()

from elasticdl.python.elasticdl.evaluator.evaluator import Evaluator


def _pos_int(arg):
    res = int(arg)
    if res <= 0:
        raise ValueError("Positive integer argument required. Got %s" % res)
    return res

def _parse_args():
    parser = argparse.ArgumentParser(description="ElasticDL Evaluator")
    parser.add_argument(
        "--model_file",
        help="Full file path of user defined neural model",
        required=True,
    )
    parser.add_argument(
        "--trained_model",
        help="The trained model file path produced by ElasticDL training job",
        required=True,
    )
    parser.add_argument(
        "--data_dir",
        help="The data directory for evaluation",
        required=True,
    )
    parser.add_argument(
        "--codec_type",
        default="bytes",
        choices=["tf_example", "bytes"],
        help="Type of codec(tf_example or bytes)",
    )
    parser.add_argument(
        "--minibatch_size",
        type=_pos_int,
        help="Minibatch size used by evaluator to compute metrics",
        default='10',
    )
    parser.add_argument(
        "--log_level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        type=str.upper,
        default='WARNING',
        help="Set the logging level",
    )

    return parser.parse_args()

def main():
    args = _parse_args()

    # Initialize logger
    logging.basicConfig(
        format='%(asctime)s %(name)s %(levelname)-8s '
        '[%(filename)s:%(lineno)d] %(message)s',
    )
    logging.getLogger().setLevel(args.log_level)

    evaluator = Evaluator(
        args.model_file,
        args.trained_model,
        args.data_dir,
        codec_type=args.codec_type,
        batch_size=args.minibatch_size,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
