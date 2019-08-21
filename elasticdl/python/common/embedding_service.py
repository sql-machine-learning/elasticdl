import argparse
import subprocess
import time

import numpy as np
from rediscluster import RedisCluster

from elasticdl.python.common import k8s_client as k8s
from elasticdl.python.common.args import pos_int
from elasticdl.python.common.log_util import default_logger as logger


class EmbeddingService(object):
    """Redis implementation of EmbeddingService"""

    def __init__(self, embedding_endpoint=None, replicas=1):
        """
        Arguments:
            embedding_endpoint: The address(ip/url) and service's port map for
            Redis cluster.
            {
                address0: [port list],
                address1: [port list],
                ...
            }
            replicas: Number of slaves per redis master

        Logic of starting embedding service :
        master.main   EmbeddingService    k8s_client
          |                 |                |
          1 --------------> 2  ----------->  |
          |                 |                3
          5 <-------------- 4  <----------   |


        1. master.main calls EmbeddingService.start_embedding_service
           when the embedding service is required by the model.
        2. EmbeddingService.start_embedding_service calls
           EmbeddingService.start_embedding_pod_and_redis to ask
           k8s_client create pods for Redis.
        3. k8s_client creates pods, then pods call
           EmbeddingService.start_redis_service() to start their local
           redis instances.
        4. After pods running, EmbeddingService.start_embedding_service
           gets and saves addresses(ip/dns and port) of pods, and creates a
           Redis Cluster base on these addresses.
        5. EmbeddingService.start_embedding_service returns addresses to
           master.main, master.main saves addresses for master/worker
           accessing the Redis.

        """
        self._embedding_endpoint = embedding_endpoint
        self._replicas = replicas

    def start_embedding_service(
        self,
        embedding_service_id=0,
        resource_request="cpu=1,memory=4096Mi",
        resource_limit="cpu=1,memory=4096Mi",
        pod_priority=None,
        volume=None,
        image_pull_policy=None,
        restart_policy="Never",
        **kargs,
    ):
        self.start_embedding_pod_and_redis(
            command=["python"],
            args=["-m", "elasticdl.python.common.embedding_service"],
            embedding_service_id=embedding_service_id,
            resource_request=resource_request,
            resource_limit=resource_limit,
            pod_priority=pod_priority,
            volume=volume,
            image_pull_policy=image_pull_policy,
            restart_policy=restart_policy,
            **kargs,
        )
        return self._create_redis_cluster()

    def _create_redis_cluster(self):
        redis_cluster_command = " ".join(
            [
                "%s:%d" % (ip, port)
                for ip in self._embedding_endpoint
                for port in self._embedding_endpoint[ip]
            ]
        )
        try:
            command = (
                "echo yes | redis-cli --cluster create %s "
                "--cluster-replicas %d"
                % (redis_cluster_command, self._replicas)
            )
            redis_process = subprocess.Popen(
                [command], shell=True, stdout=subprocess.DEVNULL
            )
            redis_process.wait()
        except Exception as e:
            logger.error(e)
            return None
        else:
            return self._embedding_endpoint

    def stop_embedding_service(self, save="nosave"):
        for redis_node in [
            "-h %s -p %d" % (ip, port)
            for ip in self._embedding_endpoint
            for port in self._embedding_endpoint[ip]
        ]:
            try:
                command = "redis-cli %s shutdown %s" % (redis_node, save)
                redis_process = subprocess.Popen(
                    [command], shell=True, stdout=subprocess.DEVNULL
                )
                redis_process.wait()
            except Exception as e:
                logger.error(e)
                return False

        return True

    def _get_embedding_cluster(self):
        startup_nodes = [
            {"host": ip, "port": "%d" % (port)}
            for ip in self._embedding_endpoint
            for port in self._embedding_endpoint[ip]
        ]
        try:
            redis_cluster = RedisCluster(
                startup_nodes=startup_nodes, decode_responses=False
            )
        except Exception as e:
            logger.error(e)
            return None
        else:
            return redis_cluster

    def _parse_embedding_service_args(self):
        parser = argparse.ArgumentParser(description="Embedding Service")
        parser.add_argument(
            "--first_port",
            default=30001,
            type=pos_int,
            help="The first listening port of embedding service",
        )
        parser.add_argument(
            "--num_of_redis_instances",
            default=6,
            type=pos_int,
            help="The number of redis instances",
        )
        parser.add_argument(
            "--cluster_node_timeout",
            default=2000,
            type=pos_int,
            help="The maximum amount of time a Redis Cluster node "
            "can be unavailable",
        )

        args = parser.parse_args()

        return args

    def start_redis_service(self):
        args = self._parse_embedding_service_args()
        logger.info(
            "Starting redis server on ports: %d - %d, "
            "--cluster_node_timeout %d"
            % (
                args.first_port,
                args.first_port + args.num_of_redis_instances - 1,
                args.cluster_node_timeout,
            )
        )
        for i in range(args.num_of_redis_instances):
            port = args.first_port + i
            command = (
                "redis-server --port %d --cluster-enabled yes "
                "--cluster-config-file nodes-%d.conf --cluster-node-timeout"
                " %d --appendonly yes --appendfilename appendonly-%d.aof "
                "--dbfilename dump-%d.rdb --logfile %d.log --daemonize yes "
                "--protected-mode no"
                % (port, port, args.cluster_node_timeout, port, port, port)
            )
            redis_process = subprocess.Popen(
                [command], shell=True, stdout=subprocess.DEVNULL
            )
            redis_process.wait()

    # TODO: Now, we use single pod to start redis cluster service, we
    # should support a redis cluster service running on multi-pods in
    # the future.
    def start_embedding_pod_and_redis(
        self,
        command,
        args,
        embedding_service_id=0,
        resource_request="cpu=1,memory=4096Mi",
        resource_limit="cpu=1,memory=4096Mi",
        pod_priority=None,
        volume=None,
        image_pull_policy=None,
        restart_policy="Never",
        **kargs,
    ):
        logger.info("Starting pod for embedding service ...")
        self._k8s_client = k8s.Client(event_callback=None, **kargs)
        pod = self._k8s_client.create_embedding_service(
            worker_id=embedding_service_id,
            resource_requests=resource_request,
            resource_limits=resource_limit,
            pod_priority=pod_priority,
            volume=volume,
            image_pull_policy=image_pull_policy,
            command=command,
            args=args,
            restart_policy=restart_policy,
        )

        # TODO: assign address with pod's domain name instead of pod's ip.
        # and should not fix ports
        address_ip = pod.status.pod_ip
        while not address_ip:
            pod = self._k8s_client.get_embedding_service_pod(
                embedding_service_id
            )
            address_ip = pod.status.pod_ip
        self._embedding_endpoint = {address_ip: [30001 + i for i in range(6)]}

    @staticmethod
    def lookup_embedding(
        keys=None, embedding_endpoint=None, parse_type=np.float64
    ):
        if not embedding_endpoint:
            raise Exception("Can't found embedding service!")
        if not keys:
            return None
        startup_nodes = [
            {"host": ip, "port": "%d" % (port)}
            for ip in embedding_endpoint
            for port in embedding_endpoint[ip]
        ]
        embedding_vector = []
        embedding_service = RedisCluster(
            startup_nodes=startup_nodes, decode_responses=False
        ).pipeline()
        for key in keys:
            embedding_service.get(key)
        output_dim = len(keys)
        embedding_vector = [
            np.frombuffer(vector, parse_type)
            for vector in embedding_service.execute()
        ]
        embedding_vector = np.concatenate(embedding_vector, axis=0)
        embedding_vector = embedding_vector.reshape((output_dim, -1))
        return embedding_vector

    @staticmethod
    def update_embedding(
        keys=None,
        embedding_vectors=None,
        embedding_endpoint=None,
        nx_flag=False,
    ):
        if not embedding_endpoint:
            raise Exception("Can't find embedding service!")
        if (
            keys is None
            or embedding_vectors is None
            or len(keys) != embedding_vectors.shape[0]
        ):
            raise Exception(
                "keys and embedding_vectors can not be None. "
                "And the length of keys must equal to the first dimension "
                "of embedding_vectors's shape."
            )
        startup_nodes = [
            {"host": ip, "port": "%d" % (port)}
            for ip in embedding_endpoint
            for port in embedding_endpoint[ip]
        ]
        embedding_service = RedisCluster(
            startup_nodes=startup_nodes, decode_responses=False
        ).pipeline()
        for index, key in enumerate(keys):
            embedding_service.set(
                key, embedding_vectors[index].tobytes(), nx=nx_flag
            )
        embedding_service.execute()


if __name__ == "__main__":
    EmbeddingService().start_redis_service()

    # TODO: Keep the pod running with kubernetes config
    while True:
        time.sleep(1)
