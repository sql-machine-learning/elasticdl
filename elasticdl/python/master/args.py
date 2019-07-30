import argparse

from elasticdl.python.common.args import add_common_params, add_train_params


def _pos_int(arg):
    res = int(arg)
    if res <= 0:
        raise ValueError("Positive integer argument required. Got %s" % res)
    return res


def _non_neg_int(arg):
    res = int(arg)
    if res < 0:
        raise ValueError(
            "Non-negative integer argument required. Get %s" % res
        )
    return res


def parse_args():
    parser = argparse.ArgumentParser(description="ElasticDL Master")
    parser.add_argument(
        "--port",
        default=50001,
        type=_pos_int,
        help="The listening port of master",
    )
    parser.add_argument(
        "--worker_image", help="Docker image for workers", default=None
    )
    parser.add_argument(
        "--worker_pod_priority", help="Priority requested by workers"
    )
    parser.add_argument(
        "---prediction_data_dir",
        help="Prediction data directory. Files should be in RecordIO format",
        default="",
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        default="INFO",
        help="The logging level. Default to WARNING",
    )
    add_common_params(parser)
    add_train_params(parser)

    args = parser.parse_args()

    if all(
        v == "" or v is None
        for v in [
            args.training_data_dir,
            args.evaluation_data_dir,
            args.prediction_data_dir,
        ]
    ):
        raise ValueError(
            "At least one of the data directories needs to be provided"
        )

    if args.prediction_data_dir and (
        args.training_data_dir or args.evaluation_data_dir
    ):
        raise ValueError(
            "Running prediction together with training or evaluation "
            "is not supported"
        )
    if args.prediction_data_dir and not args.checkpoint_filename_for_init:
        raise ValueError(
            "checkpoint_filename_for_init is required for running "
            "prediction job"
        )

    return args
