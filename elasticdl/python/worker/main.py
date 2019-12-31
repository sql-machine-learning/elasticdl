import grpc

from elasticdl.python.common import log_utils
from elasticdl.python.common.args import parse_worker_args
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
            # addr is in the form as "ps-pod-name.namespace.svc:port"
            channel = build_channel(addr)

            # Wait the channel is ready by a Future object.
            grpc.channel_ready_future(channel).result()
            logger.info(
                "grpc channel %s to connect pod %s is ready"
                % (addr, addr.split(".")[0])
            )
            ps_channels.append(channel)

    worker = Worker(
        args,
        channel=master_channel,
        ps_channels=ps_channels,
        set_parallelism=True,
    )
    worker.run()


if __name__ == "__main__":
    main()
