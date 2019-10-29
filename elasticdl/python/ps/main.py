from elasticdl.python.common.args import parse_ps_args
from elasticdl.python.ps.parameter_server import ParameterServer


def main():
    args = parse_ps_args()
    pserver = ParameterServer(args)
    pserver.prepare()
    pserver.run()


if __name__ == "__main__":
    main()
