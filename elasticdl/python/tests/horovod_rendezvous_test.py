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

import unittest

from elasticdl.python.master.rendezvous_server import HorovodRendezvousServer


class HorovodRendezvousServerTest(unittest.TestCase):
    def setUp(self):
        self.rendezvous_server = HorovodRendezvousServer(
            server_host="127.0.0.1"
        )
        self.rendezvous_server.start()

    def test_get_host_plan(self):
        self.rendezvous_server._cur_rendezvous_hosts = [
            "127.0.0.2",
            "127.0.0.3",
        ]
        host_alloc_plan = self.rendezvous_server._get_host_plan()
        self.assertEqual(host_alloc_plan[0].hostname, "127.0.0.2")
        self.assertEqual(host_alloc_plan[0].rank, 0)
        self.assertEqual(host_alloc_plan[0].size, 2)
        self.assertEqual(host_alloc_plan[1].hostname, "127.0.0.3")
        self.assertEqual(host_alloc_plan[1].rank, 1)
        self.assertEqual(host_alloc_plan[1].size, 2)

    def test_set_worker_hosts(self):
        self.rendezvous_server.add_worker("127.0.0.2")
        self.rendezvous_server.add_worker("127.0.0.3")
        rank_0 = self.rendezvous_server.get_worker_host_rank("127.0.0.2")
        rank_1 = self.rendezvous_server.get_worker_host_rank("127.0.0.3")
        self.assertEqual(rank_0, 0)
        self.assertEqual(rank_1, 1)
        self.assertEqual(
            self.rendezvous_server._cur_rendezvous_completed, True
        )
        self.assertEqual(self.rendezvous_server._rendezvous_id, 1)

        self.rendezvous_server.remove_worker("127.0.0.2")
        self.rendezvous_server.add_worker("127.0.0.1")
        self.rendezvous_server._init_rendezvous_server()
        self.assertEqual(self.rendezvous_server._rendezvous_id, 2)

    def test_get_attr(self):
        self.rendezvous_server.add_worker("127.0.0.2")
        self.rendezvous_server.add_worker("127.0.0.3")
        self.assertEqual(
            self.rendezvous_server.get_rendezvous_host(), "127.0.0.1"
        )
        self.assertEqual(
            self.rendezvous_server.get_worker_host_rank("127.0.0.2"), 0
        )
        self.assertEqual(self.rendezvous_server.get_size(), 2)
        self.assertEqual(self.rendezvous_server.get_rendezvous_id(), 1)


if __name__ == "__main__":
    unittest.main()
