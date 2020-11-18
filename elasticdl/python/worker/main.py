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

from elasticdl.python.common import log_utils
from elasticdl.python.common.args import parse_worker_args
from elasticdl.python.common.grpc_utils import build_channel
from elasticdl.python.worker.master_client import MasterClient
from elasticdl.python.worker.ps_client import build_ps_client
from elasticdl.python.worker.worker import Worker
from elasticdl_client.common.constants import DistributionStrategy


def main():
    args = parse_worker_args()
    logger = log_utils.get_logger(__name__)
    logger.info("Starting worker %d", args.worker_id)
    if args.master_addr is None:
        raise ValueError("master_addr is missing for worker")

    master_client = MasterClient(
        build_channel(args.master_addr), args.worker_id
    )

    ps_client = (
        build_ps_client(args.ps_addrs, logger)
        if args.distribution_strategy == DistributionStrategy.PARAMETER_SERVER
        else None
    )

    worker = Worker(
        args,
        master_client=master_client,
        ps_client=ps_client,
        set_parallelism=True,
    )
    worker.run()


if __name__ == "__main__":
    main()
