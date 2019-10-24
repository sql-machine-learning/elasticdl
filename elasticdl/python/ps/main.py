from elasticdl.python.common.args import parse_ps_args
from elasticdl.python.common.log_utils import get_logger
from elasticdl.python.common.model_utils import (
    get_module_file_path,
    load_module,
)


def main():
    args = parse_ps_args()
    logger = get_logger("PS", level=args.log_level.upper())

    model_module = load_module(
        get_module_file_path(args.model_zoo, args.model_def)
    ).__dict__
    optimizer = model_module[args.optimizer]()

    logger.info("Starting PS pod with optimizer as %s", optimizer._name)
    # TODO: create `Parameters` class instance, and start PS RPC servicer.
