import grpc

from elasticdl.python.common import log_utils
from elasticdl.python.common.args import parse_worker_args
from elasticdl.python.common.constants import GRPC
from elasticdl.python.worker.worker import Worker


def main():
    args = parse_worker_args()
    if args.master_addr is None:
        raise ValueError("master_addr is missing for worker")
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

    # TODO, create PS channels here
    ps_addrs = args.ps_addrs.split(",")
    # Just print ps_addrs out to avoid flake8 failure
    # This print can be removed once we initialize ps_channels
    # by using ps_addrs
    print("Parameter server addresses are %s" % ps_addrs)
    ps_channels = None

    logger = log_utils.get_logger(__name__)

    logger.info("Starting worker %d", args.worker_id)
    worker = Worker(args, channel=channel, ps_channels=ps_channels)
    worker.run()


if __name__ == "__main__":
    main()
