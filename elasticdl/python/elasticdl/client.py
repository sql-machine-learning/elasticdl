import argparse

from elasticdl.python.common.args import (
    add_common_params,
    add_evaluate_params,
    add_train_params,
)
from elasticdl.python.elasticdl.api import evaluate, train


def main():
    parser = argparse.ArgumentParser(
        usage="""elasticdl <command> [<args>]

Below is the list of supported commands:
train         Submit a ElasticDL distributed training job.
evaluate      Submit a ElasticDL distributed evaluation job.
"""
    )
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train", help="elasticdl train -h")
    train_parser.set_defaults(func=train)
    add_common_params(train_parser)
    add_train_params(train_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="elasticdl evaluate -h"
    )
    evaluate_parser.set_defaults(func=evaluate)
    add_common_params(evaluate_parser)
    add_evaluate_params(evaluate_parser)

    args, argv = parser.parse_known_args()
    args.func(args)


if __name__ == "__main__":
    main()
