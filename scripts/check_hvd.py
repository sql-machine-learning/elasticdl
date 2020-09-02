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


try:
    from horovod.runner.common.util.hosts import (
        get_host_assignments,
        parse_hosts,
    )

    print(
        "Found hosts.get_host_assignments",
        str(get_host_assignments),
        str(parse_hosts),
    )
except ImportError:
    print("Cannot find hosts.get_host_assignments")

try:
    from horovod.runner.http.http_server import RendezvousServer

    print("Found http_server.RendezvousServer", str(RendezvousServer))
except ImportError:
    print("Cannot find http_server.RendezvousServer")
