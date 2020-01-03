import time
from concurrent import futures

import grpc
from kubernetes import client, config

from elasticdl.proto import elasticdl_pb2_grpc
from elasticdl.python.common.constants import GRPC, PodStatus
from elasticdl.python.common.grpc_utils import build_channel
from elasticdl.python.common.k8s_client import get_master_pod_name
from elasticdl.python.common.log_utils import get_logger
from elasticdl.python.common.lr_scheduler import add_lr_scheduler_to_optimizer
from elasticdl.python.common.model_utils import (
    get_module_file_path,
    load_module,
)
from elasticdl.python.common.save_utils import CheckpointSaver
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
        self._set_lr_scheduler(model_module, args.learning_rate_scheduler)
        self.ps_id = args.ps_id
        self.num_ps_pods = args.num_ps_pods
        self.num_workers = args.num_workers
        # Create Parameters instance
        self.parameters = Parameters()
        if args.master_addr is None:
            raise ValueError("master_addr is missing for parameter servers")
        self.master_channel = build_channel(args.master_addr)
        self.evaluation_steps = args.evaluation_steps

        self.master_name = get_master_pod_name(args.job_name)
        self.namespace = args.namespace
        self._init_checkpoint_saver(args)
        self._restore_params_from_checkpoint(args.checkpoint_dir_for_init)
        self._debug_info_needed = args.log_level.upper() == "DEBUG"

    def _set_lr_scheduler(self, model_module, learning_rate_scheduler_arg):
        if learning_rate_scheduler_arg in model_module:
            self.lr_scheduler = add_lr_scheduler_to_optimizer(
                self.optimizer, model_module[learning_rate_scheduler_arg]
            )
        else:
            self.lr_scheduler = None

    def _restore_params_from_checkpoint(self, checkpoint_dir_for_init):
        """Restore parameters from a checkpint directory for the PS instance
        """
        if not checkpoint_dir_for_init:
            self.logger.info("checkpoint directory for init is None")
            return

        if not CheckpointSaver.check_checkpoint_valid(checkpoint_dir_for_init):
            raise ValueError("Invalid checkpoint directory")

        self.parameters = CheckpointSaver.restore_params_from_checkpoint(
            checkpoint_dir_for_init, self.ps_id, self.num_ps_pods
        )
        self.parameters.init_status = True
        self.logger.info(
            "The version of restored parameters is %d"
            % self.parameters.version
        )

    def _init_checkpoint_saver(self, args):
        if all(
            [
                args.checkpoint_dir,
                args.checkpoint_steps,
                args.keep_checkpoint_max,
            ]
        ):
            self.checkpoint_saver = CheckpointSaver(
                args.checkpoint_dir,
                args.checkpoint_steps,
                args.keep_checkpoint_max,
                include_evaluation=False,
            )
        else:
            self.checkpoint_saver = None
            self.logger.warning(
                "Invalid checkpoint config and no model will be saved"
            )

    def prepare(self):
        max_workers = min(self.num_workers, 64)
        self.logger.info("The max threads in PS servers is %d" % max_workers)
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
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
            self.lr_scheduler,
            lr_staleness_modulation=self.lr_staleness_modulation,
            use_async=self.use_async,
            evaluation_steps=self.evaluation_steps,
            master_channel=self.master_channel,
            checkpoint_saver=self.checkpoint_saver,
            ps_id=self.ps_id,
            num_ps_pods=self.num_ps_pods,
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
                if master_pod.status.phase == PodStatus.SUCCEEDED:
                    self.logger.info("Master pod is Succeeded")
                    break
                elif master_pod.status.phase == PodStatus.FAILED:
                    self.logger.info("Master pod is Failed")
                    break
                elif (
                    master_pod.status.phase == PodStatus.RUNNING
                    and master_pod.metadata.labels["status"]
                    == PodStatus.FINISHED
                ):
                    self.logger.info(
                        "Task is finished, "
                        "master pod is still running tensorboard service"
                    )
                    break

                if self._debug_info_needed:
                    self.logger.debug(
                        "Parameters info:\n%s" % self.parameters.debug_info()
                    )
        except KeyboardInterrupt:
            self.logger.warning("Server stopping")

        self.server.stop(0)
        self.logger.info("RPC server stopped")
