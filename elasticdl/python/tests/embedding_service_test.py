import subprocess
import time
import unittest

from elasticdl.python.common.embedding_service import EmbeddingService


def start_redis_instances():
    for i in range(6):
        port = 33001 + i
        embedding_process = subprocess.Popen(
            [
                "redis-server --port %d --cluster-enabled yes "
                "--cluster-config-file nodes-%d.conf "
                "--cluster-node-timeout 200 --appendonly yes --appendfilename "
                "appendonly-%d.aof --dbfilename dump-%d.rdb "
                "--logfile %d.log --daemonize yes --protected-mode no"
                % (port, port, port, port, port)
            ],
            shell=True,
            stdout=subprocess.DEVNULL,
        )
        embedding_process.wait()

    embedding_endpoint = {"127.0.0.1": [33001 + i for i in range(6)]}
    return embedding_endpoint


class EmbeddingServiceTest(unittest.TestCase):
    def test_embedding_service(self):
        embedding_endpoint = start_redis_instances()
        # start
        embedding_service = EmbeddingService(embedding_endpoint)
        embedding_endpoint = embedding_service._create_redis_cluster()
        self.assertFalse(embedding_endpoint is None)
        # connection
        redis_cluster = embedding_service._get_embedding_cluster()
        # check status of cluster
        time.sleep(1)
        self.assertFalse(redis_cluster is None)
        # set value to a key
        self.assertTrue(redis_cluster.set("test_key", "OK", nx=True))
        # set value to a key existed
        self.assertTrue(redis_cluster.set("test_key", "OK", nx=True) is None)
        self.assertEqual(b"OK", redis_cluster.get("test_key"))
        # close
        self.assertTrue(embedding_service.stop_embedding_service())


if __name__ == "__main__":
    unittest.main()
