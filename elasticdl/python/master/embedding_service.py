import numpy as np
from rediscluster import RedisCluster

from elasticdl.python.common import k8s_client as k8s
from elasticdl.python.common.log_util import default_logger as logger
from elasticdl.python.master.redis_util import run_shell_command


class EmbeddingService(object):
    """Redis implementation of EmbeddingService"""

    def __init__(self, replicas=1):
        """
        Arguments:
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
           EmbeddingService._start_embedding_pod_and_redis to ask
           k8s_client create pods for Redis.
        3. k8s_client creates pods, then pods call
           redis_util.start_redis_service to start their local
           redis instances.
        4. After pods running, EmbeddingService.start_embedding_service
           gets and saves addresses(ip/dns and port) of pods, and creates a
           Redis Cluster base on these addresses.
        5. EmbeddingService.start_embedding_service returns addresses to
           master.main, master.main saves addresses for master/worker
           accessing the Redis.

        """
        self._embedding_service_endpoint = None
        self._replicas = replicas

        # TODO: Now, we use a single pod to start redis cluster service, we
        # should support a redis cluster service running on multi-pods in
        # the future.

    def _start_embedding_pod_and_redis(
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
        # and should not use fixed ports
        address_ip = pod.status.pod_ip
        while not address_ip:
            pod = k8s_client.get_embedding_service_pod(embedding_service_id)
            address_ip = pod.status.pod_ip
        self._embedding_service_endpoint = {
            address_ip: [30001 + i for i in range(6)]
        }

    def _create_redis_cluster(self, test_endpoint):
        if not self._embedding_service_endpoint and test_endpoint:
            self._embedding_service_endpoint = test_endpoint
        else:
            raise ValueError("embedding service endpoint is not set!")

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
        return_code = run_shell_command(command)
        if return_code:
            raise Exception(
                "Create Redis cluster failed with command: %s" % command
            )

        startup_nodes = [
            {"host": ip, "port": "%d" % port}
            for ip, port_list in self._embedding_service_endpoint.items()
            for port in port_list
        ]
        try:
            self._redis_cluster = RedisCluster(
                startup_nodes=startup_nodes, decode_responses=False
            )
        except Exception as e:
            logger.error(e)
            return None

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
        self._start_embedding_pod_and_redis(
            command=["python"],
            args=["-m", "elasticdl.python.master.redis_util"],
            embedding_service_id=embedding_service_id,
            resource_request=resource_request,
            resource_limit=resource_limit,
            pod_priority=pod_priority,
            volume=volume,
            image_pull_policy=image_pull_policy,
            restart_policy=restart_policy,
            **kargs,
        )
        self._create_redis_cluster(test_endpoint=None)

    def stop_embedding_service(self, save="nosave"):
        failed_redis_nodes = []
        for redis_node in [
            "-h %s -p %d" % (ip, port)
            for ip, port_list in self._embedding_service_endpoint.items()
            for port in port_list
        ]:
            command = "redis-cli %s shutdown %s" % (redis_node, save)
            return_code = run_shell_command(command)
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

    def lookup_embedding(self, keys=None, parse_type=np.float32):
        """
        Arguments:
            keys: The list of key, which be used to locate embedding vector
            parse_type: The type of saved data.

        Returns:
            A tuple contains embedding_vectors and unknown_keys_idx.
            embedding_vectors: A list of lookup's result, ndarray of
            embedding vector for found, `None` for embedding vector not found
            unknown_keys_idx: If some keys do not have a corresponding
            embedding vector, it returns the index of these keys.
        """
        if keys is None:
            return [], []
        embedding_service = self._redis_cluster.pipeline()
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

    def update_embedding(
        self, keys=None, embedding_vectors=None, set_if_not_exist=False
    ):
        """
        Arguments:
            keys: The list of key, which be used to locate embedding vector.
            embedding_vectors: The embedding vectors.
            service parse_type: The type of saved data.
            set_if_not_exist: If this argument is `True`, it will set embedding
                vector only when this embedding vector doesn't exist.
        Returns:
            None
        """
        if keys is None:
            keys = []
        if embedding_vectors is None:
            embedding_vectors = []
        key_num = len(keys)
        embedding_vector_num = len(embedding_vectors)
        if key_num != embedding_vector_num:
            raise Exception(
                "The number of keys %d does not equal to the number of "
                "embedding vectors %d." % (key_num, embedding_vector_num)
            )
        if not key_num:
            return
        embedding_service = self._redis_cluster.pipeline()
        for index, key in enumerate(keys):
            embedding_service.set(
                key, embedding_vectors[index].tobytes(), nx=set_if_not_exist
            )
        embedding_service.execute()
