import socket
import subprocess
import tempfile
import time
import unittest

import numpy as np

from elasticdl.python.common.embedding_service import EmbeddingService


def get_free_port():
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = s.getsockname()[1]
            # Ports with numbers 0-1023: system or well-known ports
            # Ports with numbers 1024-49151: user or registered ports
            # Ports with numbers 49152-65535: dynamic and/or private ports
            if port < 49152 and port > 1023:
                return port


def start_redis_instances(temp_dir):
    port_list = []
    while len(port_list) < 6:
        port = get_free_port()
        embedding_process = subprocess.Popen(
            [
                "redis-server --port %d --dir %s --cluster-enabled yes "
                "--cluster-config-file nodes-%d.conf "
                "--cluster-node-timeout 200 --appendonly yes --appendfilename "
                "appendonly-%d.aof --dbfilename dump-%d.rdb "
                "--logfile %d.log --daemonize yes --protected-mode no"
                % (port, temp_dir, port, port, port, port)
            ],
            shell=True,
            stdout=subprocess.DEVNULL,
        )
        embedding_process.wait()
        if not embedding_process.returncode:
            port_list.append(port)
        else:
            embedding_process.kill()

    embedding_endpoint = {"127.0.0.1": port_list}
    print(port_list)
    return embedding_endpoint


class EmbeddingServiceTest(unittest.TestCase):
    def test_embedding_service(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            embedding_endpoint = start_redis_instances(temp_dir)
            # start
            embedding_service = EmbeddingService(embedding_endpoint)
            embedding_endpoint = embedding_service._create_redis_cluster()
            # wait for cluster up-running
            time.sleep(1)
            self.assertFalse(embedding_endpoint is None)
            # connection
            redis_cluster = embedding_service._get_embedding_cluster()
            self.assertFalse(redis_cluster is None)
            # set value to a key
            self.assertTrue(redis_cluster.set("test_key", "OK", nx=True))
            # set value to a key existed
            self.assertTrue(
                redis_cluster.set("test_key", "OK", nx=True) is None
            )
            self.assertEqual(b"OK", redis_cluster.get("test_key"))
            # close
            self.assertTrue(embedding_service.stop_embedding_service())

    def test_lookup_and_update_embedding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            embedding_endpoint = start_redis_instances(temp_dir)
            # start
            embedding_service = EmbeddingService(embedding_endpoint)
            embedding_endpoint = embedding_service._create_redis_cluster()
            # wait for cluster up-running
            time.sleep(1)
            origin_data = np.random.rand(100, 10).astype(np.float32)
            keys = ["test_%d" % i for i in range(origin_data.shape[0])]

            EmbeddingService.update_embedding(
                keys, origin_data, embedding_endpoint
            )
            lookup_data, unknown_keys_idx = EmbeddingService.lookup_embedding(
                keys, embedding_endpoint, parse_type=np.float32
            )
            self.assertTrue(len(unknown_keys_idx) == 0)
            output_length = len(keys)
            lookup_data = np.concatenate(lookup_data, axis=0)
            lookup_data = lookup_data.reshape((output_length, -1))
            self.assertTrue(np.equal(origin_data, lookup_data).all())

            # Test set_if_not_exist
            origin_data_2 = np.random.rand(100, 10).astype(np.float32)
            self.assertFalse(np.equal(origin_data, origin_data_2).all())
            EmbeddingService.update_embedding(
                keys, origin_data_2, embedding_endpoint, set_if_not_exist=True
            )
            lookup_data, unknown_keys_idx = EmbeddingService.lookup_embedding(
                keys, embedding_endpoint, parse_type=np.float32
            )
            lookup_data = np.concatenate(lookup_data, axis=0)
            lookup_data = lookup_data.reshape((output_length, -1))
            self.assertTrue(np.equal(origin_data, lookup_data).all())
            self.assertFalse(np.equal(origin_data_2, lookup_data).all())

            # Test non-exist keys
            keys_do_not_exist = ["test_no_exist_%d" % i for i in range(10)]
            lookup_data, unknown_keys_idx = EmbeddingService.lookup_embedding(
                keys_do_not_exist, embedding_endpoint, parse_type=np.float32
            )
            self.assertTrue(len(unknown_keys_idx) == 10)
            self.assertTrue(len(lookup_data) == 10)
            # Close
            self.assertTrue(embedding_service.stop_embedding_service())


if __name__ == "__main__":
    unittest.main()
