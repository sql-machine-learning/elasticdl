import argparse
import os
import subprocess
import time

from elasticdl.python.common.args import pos_int
from elasticdl.python.common.constants import Redis
from elasticdl.python.common.log_util import default_logger as logger


def run_shell_command(command):
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


def parse_embedding_service_args():
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


def start_redis_service():
    args = parse_embedding_service_args()
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
        return_code = run_shell_command(command)
        if return_code:
            failed_port.append(port)
    if failed_port:
        local_ip = os.getenv("MY_POD_IP", "localhost")
        logger.info(
            "%s starts these redis instances failed: %s"
            % (local_ip, ";".join(map(str, failed_port)))
        )


if __name__ == "__main__":

    start_redis_service()
    # TODO: Keep the pod running with kubernetes config
    while True:
        time.sleep(1)
