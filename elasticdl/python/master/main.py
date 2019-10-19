from elasticdl.python.common.args import parse_master_args
from elasticdl.python.common.log_utils import get_logger
from elasticdl.python.master.master import Master


def main():
    args = parse_master_args()
    Master.logger = get_logger("master", level=args.log_level.upper())
    master = Master(args)
    master.start(args)
    master.run()


if __name__ == "__main__":
    main()
