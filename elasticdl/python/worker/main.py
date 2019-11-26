import grpc

from elasticdl.python.common import log_utils
from elasticdl.python.common.args import parse_worker_args
from elasticdl.python.common.constants import GRPC
from elasticdl.python.common.grpc_utils import build_channel
from elasticdl.python.worker.worker import Worker


def main():
    args = parse_worker_args()
    logger = log_utils.get_logger(__name__)
    logger.info("Starting worker %d", args.worker_id)
    if args.master_addr is None:
        raise ValueError("master_addr is missing for worker")

    master_channel = build_channel(args.master_addr)

    ps_channels = []
    if args.ps_addrs:
        ps_addrs = args.ps_addrs.split(",")
        for addr in ps_addrs:
            channel = grpc.insecure_channel(
                addr,
                options=[
                    (
                        "grpc.max_send_message_length",
                        GRPC.MAX_SEND_MESSAGE_LENGTH,
                    ),
                    (
                        "grpc.max_receive_message_length",
                        GRPC.MAX_RECEIVE_MESSAGE_LENGTH,
                    ),
                ],
            )

            # Wait the channel is ready by a Future object.
            grpc.channel_ready_future(channel).result()
            logger.info("grpc channel to %s is ready" % addr)
            ps_channels.append(channel)

    worker = Worker(args, channel=master_channel, ps_channels=ps_channels)
    worker.run()


if __name__ == "__main__":
    main()
