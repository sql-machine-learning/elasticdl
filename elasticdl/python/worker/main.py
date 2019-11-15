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
        # TODO: use ps_addrs from master directly after ps service is working.
        #       Get ps pod ip for ps grpc connection for now.
        ps_addrs = args.ps_addrs.split(",")
        from kubernetes import client, config
        import time

        config.load_incluster_config()
        api = client.CoreV1Api()

        for addr in ps_addrs:
            # addr is in the form as "ps-pod-name.namespace.svc:port"
            addr_splitted = addr.split(".")
            while True:
                pod = api.read_namespaced_pod(
                    namespace=addr_splitted[1], name=addr_splitted[0]
                )
                if pod.status.pod_ip:
                    break
                # If ps pod is not ready yet, sleep 2 seconds and try again.
                time.sleep(2)
            addr = pod.status.pod_ip + ":" + addr.split(":")[-1]
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
            ps_channels.append(channel)

    worker = Worker(args, channel=master_channel, ps_channels=ps_channels)
    worker.run()


if __name__ == "__main__":
    main()
