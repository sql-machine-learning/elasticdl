import time
from concurrent import futures

import grpc
from kubernetes import client, config

from elasticdl.proto import elasticdl_pb2_grpc
from elasticdl.python.common.constants import GRPC
from elasticdl.python.common.k8s_client import get_master_pod_name
from elasticdl.python.common.log_utils import get_logger
from elasticdl.python.common.model_utils import (
    get_module_file_path,
    load_module,
)
from elasticdl.python.ps.parameters import Parameters
from elasticdl.python.ps.servicer import PserverServicer


class ParameterServer(object):
    def __init__(self, args):
        self.logger = get_logger("PS", level=args.log_level.upper())

        self.grads_to_wait = args.grads_to_wait
        self.lr_staleness_modulation = args.lr_staleness_modulation
        self.use_async = args.use_async
        self.port = args.port
        model_module = load_module(
            get_module_file_path(args.model_zoo, args.model_def)
        ).__dict__
        self.optimizer = model_module[args.optimizer]()
        # Create Parameters instance
        self.parameters = Parameters()

        self.master_name = get_master_pod_name(args.job_name)
        self.namespace = args.namespace

    def prepare(self):
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=64),
            options=[
                ("grpc.max_send_message_length", GRPC.MAX_SEND_MESSAGE_LENGTH),
                (
                    "grpc.max_receive_message_length",
                    GRPC.MAX_RECEIVE_MESSAGE_LENGTH,
                ),
            ],
        )
        pserver_servicer = PserverServicer(
            self.parameters,
            self.grads_to_wait,
            self.optimizer,
            lr_staleness_modulation=self.lr_staleness_modulation,
            use_async=self.use_async,
        )
        elasticdl_pb2_grpc.add_PserverServicer_to_server(
            pserver_servicer, server
        )
        server.add_insecure_port("[::]:{}".format(self.port))
        server.start()
        self.server = server
        self.logger.info("RPC Server started at port: %d", self.port)

    def run(self):
        config.load_incluster_config()
        api = client.CoreV1Api()
        try:
            while True:
                time.sleep(30)
                master_pod = api.read_namespaced_pod(
                    namespace=self.namespace, name=self.master_name
                )
                if master_pod.status.phase == "Succeeded":
                    break
        except KeyboardInterrupt:
            self.logger.warning("Server stopping")

        self.server.stop(0)
        self.logger.info("RPC server stopped")
