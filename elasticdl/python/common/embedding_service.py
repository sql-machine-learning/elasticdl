import argparse
import subprocess
import time

from rediscluster import StrictRedisCluster

from elasticdl.python.common import k8s_client as k8s
from elasticdl.python.common.log_util import default_logger as logger


class EmbeddingService(object):
    """Redis implementation of EmbeddingService"""

    def __init__(self, redis_address_map=None, replicas=1):
        """
        Arguments:
            redis_address_map: The address(ip/url) and service's port map for
            Redis cluster.
            {
                address0:[port list],
                address1:[port list],
                ...
            }

        Logic of starting embedding service :
        main       EmbeddingService      k8s_client
          |                 |                |
          1 --------------> 2  ----------->  |
          |                 |                3
          5 <-------------- 4  <----------   |


        1:Main need embedding service, asks EmbeddingService to start
          embedding service (function: start_embedding_service ).
        2:EmbeddingService accepts request,and ask k8s_client create
          pods for Redis.(function: start_embedding_pod_and_redis).
        3:k8s_client creates pods(function:k8s_client.create_embedding_service)
          then pods call EmbeddingService.start_redis_service() to start local
          redis instances.
        4:After pods start running, EmbeddingService gets and saves
          addresses(ip/dns and port) of pods, create a Redis Cluster base on
          these addresses (function:start_embedding_service)
        5:Main save addresses for master/worker accessing the database.

        """
        self._redis_address_map = redis_address_map
        self._replicas = replicas

    def start_embedding_service(
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
        self.start_embedding_pod_and_redis(
            command=command,
            args=args,
            embedding_service_id=embedding_service_id,
            resource_request=resource_request,
            resource_limit=resource_limit,
            pod_priority=pod_priority,
            volume=volume,
            image_pull_policy=image_pull_policy,
            restart_policy=restart_policy,
            **kargs,
        )
        redis_cluster_command = " ".join(
            [
                "%s:%d" % (ip, port)
                for ip in self._redis_address_map
                for port in self._redis_address_map[ip]
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
            return self._redis_address_map

    def stop_embedding_service(self, save="nosave"):
        for redis_node in [
            "-h %s -p %d" % (ip, port)
            for ip in self._redis_address_map
            for port in self._redis_address_map[ip]
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
            for ip in self._redis_address_map
            for port in self._redis_address_map[ip]
        ]
        try:
            redis_cluster = StrictRedisCluster(
                startup_nodes=startup_nodes, decode_responses=False
            )
        except Exception as e:
            logger.error(e)
            return None
        else:
            return redis_cluster

    def _pos_int(arg):
        res = int(arg)
        if res <= 0:
            raise ValueError(
                "Positive integer argument required. Got %s" % res
            )
        return res

    def _parse_embedding_service_args(self):
        parser = argparse.ArgumentParser(description="Embedding Service")
        parser.add_argument(
            "--first_port",
            default=30001,
            type=self._pos_int,
            help="The first listening port of embedding service",
        )
        parser.add_argument(
            "--num_of_redis",
            default=6,
            type=self._pos_int,
            help="The number of redis instances",
        )
        parser.add_argument(
            "--cluster_node_timeout",
            default=2000,
            type=self._pos_int,
            help="The number of redis instances",
        )

        args = parser.parse_args()

        return args

    def start_redis_service(self):
        # pod call
        logger.info("Starting redis server ...")
        args = self._parse_embedding_service_args()

        for i in range(args.num_of_redis):
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
        """Start redis """
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
        # and should'd fix port
        address_ip = pod.status.pod_ip
        while not address_ip:
            pod = self._k8s_client.get_embedding_service_pod(
                embedding_service_id
            )
            address_ip = pod.status.pod_ip
        self._redis_address_map = {address_ip: [30001 + i for i in range(6)]}

    @staticmethod
    def lookup_embedding(**kwargs):
        pass

    @staticmethod
    def update_embedding(**kwargs):
        pass


if __name__ == "__main__":
    EmbeddingService().start_redis_service()

    # TODO: Keep the pod running with kubernetes config
    while True:
        time.sleep(1)
