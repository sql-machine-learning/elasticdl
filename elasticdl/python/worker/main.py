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

import time

import grpc

from elasticdl.python.common import log_utils
from elasticdl.python.common.args import parse_worker_args
from elasticdl.python.common.constants import DistributionStrategy
from elasticdl.python.common.grpc_utils import build_channel
from elasticdl.python.worker.worker import Worker

CONNECT_PS_MAX_RETRIES = 3
CONNECT_PS_TIMEOUT = 300
# The number of seconds we wait for allreduce strategy's
# FTLib consensus service to detect the worker pod
_ALLREDUCE_STRATEGY_WARM_UP_SECS = 20


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

    if args.distribution_strategy == DistributionStrategy.ALLREDUCE:
        logger.info(
            "Wait for %s seconds for FTLib consensus service to "
            "detect the worker pod" % str(_ALLREDUCE_STRATEGY_WARM_UP_SECS)
        )
        time.sleep(_ALLREDUCE_STRATEGY_WARM_UP_SECS)

    worker = Worker(
        args,
        channel=master_channel,
        ps_channels=ps_channels,
        set_parallelism=True,
    )
    worker.run()


if __name__ == "__main__":
    main()
