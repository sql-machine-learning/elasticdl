import sys

from elasticdl.python.common.args import parse_master_args
from elasticdl.python.master.master import Master


def main():
    args = parse_master_args()
    master = Master(args)
    master.prepare()
    return master.run()


if __name__ == "__main__":
    sys.exit(main())
