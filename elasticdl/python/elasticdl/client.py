import argparse

from elasticdl.python.common.args import (
    add_clean_params,
    add_common_params,
    add_evaluate_params,
    add_predict_params,
    add_train_params,
)
from elasticdl.python.elasticdl.api import clean, evaluate, predict, train


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")
    subparsers.required = True

    train_parser = subparsers.add_parser(
        "train", help="Submit a ElasticDL distributed training job"
    )
    train_parser.set_defaults(func=train)
    add_common_params(train_parser)
    add_train_params(train_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Submit a ElasticDL distributed evaluation job"
    )
    evaluate_parser.set_defaults(func=evaluate)
    add_common_params(evaluate_parser)
    add_evaluate_params(evaluate_parser)

    predict_parser = subparsers.add_parser(
        "predict", help="Submit a ElasticDL distributed prediction job"
    )
    predict_parser.set_defaults(func=predict)
    add_common_params(predict_parser)
    add_predict_params(predict_parser)

    clean_parser = subparsers.add_parser(
        "clean", help="Cleanup local docker images built by ElasticDL"
    )
    clean_parser.set_defaults(func=clean)
    add_clean_params(clean_parser)

    args, _ = parser.parse_known_args()
    args.func(args)


if __name__ == "__main__":
    main()
