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

    redis_address_map = {"127.0.0.1": [33001 + i for i in range(6)]}
    return redis_address_map


def create_redis_redis_cluster(redis_address_map):
    redis_cluster_command = " ".join(
        [
            "%s:%d" % (ip, port)
            for ip in redis_address_map
            for port in redis_address_map[ip]
        ]
    )
    try:
        command = (
            "echo yes | redis-cli --cluster create %s "
            "--cluster-replicas %d" % (redis_cluster_command, 1)
        )
        redis_process = subprocess.Popen(
            [command], shell=True, stdout=subprocess.DEVNULL
        )
        redis_process.wait()
    except Exception as e:
        print(e)
        return None
    else:
        return redis_address_map


class EmbeddingServiceTest(unittest.TestCase):
    def test_embedding_service(self):
        redis_address_map = start_redis_instances()
        # start
        redis_address_map = create_redis_redis_cluster(redis_address_map)
        self.assertFalse(redis_address_map is None)
        time.sleep(1)
        embedding_service = EmbeddingService(redis_address_map)
        # connection
        redis_cluster = embedding_service._get_embedding_cluster()
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
