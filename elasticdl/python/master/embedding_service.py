import argparse
import os
import subprocess
import time

import numpy as np
from rediscluster import RedisCluster

from elasticdl.python.common import k8s_client as k8s
from elasticdl.python.common.args import pos_int
from elasticdl.python.common.constants import Redis
from elasticdl.python.common.log_utils import default_logger as logger


class EmbeddingService(object):
    """Redis implementation of EmbeddingService"""

    def __init__(self, embedding_service_endpoint=None, replicas=1):
        """
        Arguments:
            embedding_service_endpoint: The address(ip/url) and service's
            port map for Redis cluster.
            {
                address0: [port list],
                address1: [port list],
                ...
            }
            replicas: Number of slaves per redis master.

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
        self._embedding_service_endpoint = embedding_service_endpoint
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
            args=["-m", "elasticdl.python.master.embedding_service"],
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
                for ip, port_list in self._embedding_service_endpoint.items()
                for port in port_list
            ]
        )
        command = (
            "echo yes | redis-cli --cluster create %s "
            "--cluster-replicas %d" % (redis_cluster_command, self._replicas)
        )
        return_code = self._run_shell_command(command)
        if not return_code:
            return self._embedding_service_endpoint
        raise Exception(
            "Create Redis cluster failed with command: %s" % command
        )

    def stop_embedding_service(self, save="nosave"):
        failed_redis_nodes = []
        for redis_node in [
            "-h %s -p %d" % (ip, port)
            for ip, port_list in self._embedding_service_endpoint.items()
            for port in port_list
        ]:
            command = "redis-cli %s shutdown %s" % (redis_node, save)
            return_code = self._run_shell_command(command)
            if return_code:
                failed_redis_nodes.append(redis_node)

        if failed_redis_nodes:
            failed_redis_nodes = [i.split(" ") for i in failed_redis_nodes]
            logger.info(
                "Stop these redis nodes failed: %s."
                % ";".join(
                    [
                        "%s:%s" % (redis_node[1], redis_node[3])
                        for redis_node in failed_redis_nodes
                    ]
                )
            )
            return False

        return True

    def _get_embedding_cluster(self):
        startup_nodes = [
            {"host": ip, "port": "%d" % port}
            for ip, port_list in self._embedding_service_endpoint.items()
            for port in port_list
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

    @staticmethod
    def _parse_embedding_service_args():
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

        args, _ = parser.parse_known_args()

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
        failed_port = []
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
            return_code = self._run_shell_command(command)
            if return_code:
                failed_port.append(port)
        if failed_port:
            local_ip = os.getenv("MY_POD_IP", "localhost")
            logger.info(
                "%s starts these redis instances failed: %s"
                % (local_ip, ";".join(map(str, failed_port)))
            )

    @staticmethod
    def _run_shell_command(command):
        retry_times = 0
        redis_process = None
        while retry_times <= Redis.MAX_COMMAND_RETRY_TIMES:
            if retry_times:
                logger.warning(
                    'Command: "%s" failed to run, retry times: %d .'
                    % (command, retry_times)
                )
            redis_process = subprocess.Popen(
                [command],
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            redis_process.wait()
            if not redis_process.returncode:
                break
            redis_process.kill()
            # Wait for retry
            time.sleep(1)
            retry_times += 1
        return redis_process.returncode

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
        k8s_client = k8s.Client(**kargs)
        pod = k8s_client.create_embedding_service(
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
            pod = k8s_client.get_embedding_service_pod(embedding_service_id)
            address_ip = pod.status.pod_ip
        self._embedding_service_endpoint = {
            address_ip: [30001 + i for i in range(6)]
        }

    @staticmethod
    def lookup_embedding(
        keys=[], embedding_service_endpoint=None, parse_type=np.float32
    ):
        """
        Arguments:
            keys: The list of key, which be used to locate embedding vector
            embedding_service_endpoint: The access endpoint of embedding
            service
            parse_type: The type of saved data.

        Returns:
            A tuple contains embedding_vectors and unknown_keys_idx.
            embedding_vectors: A list of lookup's result, ndarray of
            embedding vector for found, `None` for embedding vector not found
            unknown_keys_idx: If some keys do not have a corresponding
            embedding vector, it returns the index of these keys.
        """
        if not embedding_service_endpoint:
            raise Exception("Can't find embedding service!")
        if not keys:
            return [], []
        startup_nodes = [
            {"host": ip, "port": "%d" % port}
            for ip, port_list in embedding_service_endpoint.items()
            for port in port_list
        ]
        embedding_service = RedisCluster(
            startup_nodes=startup_nodes, decode_responses=False
        ).pipeline()
        for key in keys:
            embedding_service.get(key)
        embedding_vectors = embedding_service.execute()
        unknown_keys_idx = []
        embedding_vectors_ndarray = []
        for index, vector in enumerate(embedding_vectors):
            if vector:
                embedding_vectors_ndarray.append(
                    np.frombuffer(vector, parse_type)
                )
            else:
                embedding_vectors_ndarray.append(vector)
                unknown_keys_idx.append(index)
        return embedding_vectors_ndarray, unknown_keys_idx

    @staticmethod
    def update_embedding(
        keys=[],
        embedding_vectors=[],
        embedding_service_endpoint=None,
        set_if_not_exist=False,
    ):
        """
        Arguments:
            keys: The list of key, which be used to locate embedding vector.
            embedding_vectors: The embedding vectors.
            embedding_service_endpoint: The access endpoint of embedding
            service parse_type: The type of saved data.
            set_if_not_exist: If this argument is `True`, it will set embedding
                vector only when this embedding vector doesn't exist.
        Returns:
            None
        """
        if not embedding_service_endpoint:
            raise Exception("Can't find embedding service!")
        key_num = len(keys)
        embedding_vector_num = len(embedding_vectors)
        if key_num != embedding_vector_num:
            raise Exception(
                "The number of keys %d does not equal to the number of "
                "embedding vectors %d." % (key_num, embedding_vector_num)
            )
        if not key_num:
            return
        startup_nodes = [
            {"host": ip, "port": "%d" % port}
            for ip, port_list in embedding_service_endpoint.items()
            for port in port_list
        ]
        embedding_service = RedisCluster(
            startup_nodes=startup_nodes, decode_responses=False
        ).pipeline()
        for index, key in enumerate(keys):
            embedding_service.set(
                key, embedding_vectors[index].tobytes(), nx=set_if_not_exist
            )
        embedding_service.execute()


if __name__ == "__main__":
    EmbeddingService().start_redis_service()

    # TODO: Keep the pod running with kubernetes config
    while True:
        time.sleep(1)
