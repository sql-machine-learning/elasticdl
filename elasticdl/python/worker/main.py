# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import grpc

from elasticdl.python.common import log_utils
from elasticdl.python.common.args import parse_worker_args
from elasticdl.python.common.grpc_utils import build_channel
from elasticdl.python.worker.master_client import MasterClient
from elasticdl.python.worker.worker import Worker

CONNECT_PS_MAX_RETRIES = 3
CONNECT_PS_TIMEOUT = 300


def main():
    args = parse_worker_args()
    logger = log_utils.get_logger(__name__)
    logger.info("Starting worker %d", args.worker_id)
    if args.master_addr is None:
        raise ValueError("master_addr is missing for worker")

    master_client = MasterClient(build_channel(args.master_addr))

    ps_channels = []
    if args.ps_addrs:
        ps_addrs = args.ps_addrs.split(",")

        for addr in ps_addrs:
            # addr is in the form as "ps-pod-name.namespace.svc:port"
            channel = build_channel(addr)

            succeeded = False
            for i in range(CONNECT_PS_MAX_RETRIES):
                try:
                    grpc.channel_ready_future(channel).result(
                        timeout=CONNECT_PS_TIMEOUT
                    )
                    logger.info(
                        "grpc channel %s to connect pod %s is ready"
                        % (addr, addr.split(".")[0])
                    )
                    ps_channels.append(channel)
                    succeeded = True
                    break
                except grpc.FutureTimeoutError:
                    logger.warning(
                        "Failed to connect pod %s with %d retry"
                        % (addr.split(".")[0], i)
                    )
            if not succeeded:
                raise TimeoutError(
                    "Time out to connect pod %s with 3 retries"
                    % addr.split(".")[0]
                )

    worker = Worker(
        args,
        master_client=master_client,
        ps_channels=ps_channels,
        set_parallelism=True,
    )
    worker.run()


if __name__ == "__main__":
    main()
