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
        "--model_zoo",
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
    parser.add_argument(
        "--input_fn",
        type=str,
        default="input_fn",
        help="The name of the input function defined in the model file",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="loss",
        help="The name of the loss function defined in the model file",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="optimizer",
        help="The name of the optimizer defined in the model file",
    )
    parser.add_argument(
        "--eval_metrics_fn",
        type=str,
        default="eval_metrics_fn",
        help="The name of the evaluation metrics function defined "
        "in the model file",
    )
    parser.add_argument(
        "--model_def",
        type=str,
        required=True,
        help="The import path to the model definition function/class in the "
        'model zoo, e.g. "cifar10_subclass.cifar10_subclass.CustomModel"',
    )
    parser.add_argument(
        "--model_params",
        type=str,
        default="",
        help="The dictionary of model parameters in a string that will be "
        'used to instantiate the model, e.g. "param1=1,param2=2"',
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
        args.worker_id,
        get_model_file(args.model_zoo, args.model_def),
        channel=channel,
        input_fn=args.input_fn,
        loss=args.loss,
        optimizer=args.optimizer,
        eval_metrics_fn=args.eval_metrics_fn,
        model_def=args.model_def,
        model_params=args.model_params,
    )
    worker.run()


if __name__ == "__main__":
    main()
