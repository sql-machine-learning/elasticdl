import subprocess
import unittest

from elasticdl.python.common.embedding_service import EmbeddingService


class EmbeddingServiceTest(unittest.TestCase):
    def test_embedding_service(self):
        for i in range(6):
            port = 31006 + i
            embedding_process = subprocess.Popen(
                [
                    "redis-server --port %d --cluster-enabled yes "
                    "--cluster-config-file nodes-%d.conf --cluster-node-"
                    "timeout 200 --appendonly yes --appendfilename appendonly"
                    "-%d.aof --dbfilename dump-%d.rdb --logfile %d.log "
                    "--daemonize yes --protected-mode no"
                    % (port, port, port, port, port)
                ],
                shell=True,
                stdout=subprocess.DEVNULL,
            )
            embedding_process.wait()
        redis_address_map = {"127.0.0.1": [31006 + i for i in range(6)]}
        # start
        embedding_service = EmbeddingService(redis_address_map)
        self.assertTrue(embedding_service.start_embedding_service())
        # connection
        rc = embedding_service._get_embedding_cluster()
        self.assertFalse(rc is None)
        # set value to a key
        self.assertTrue(rc.set("test_key", "OK", nx=True))
        # set value to a key existed
        self.assertTrue(rc.set("test_key", "OK", nx=True) is None)
        self.assertEqual(b"OK", rc.get("test_key"))
        # close
        self.assertTrue(embedding_service.stop_embedding_service())


if __name__ == "__main__":
    unittest.main()
