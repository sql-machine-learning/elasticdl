import os
import time
from concurrent import futures

import grpc
from kubernetes.client import V1EnvVar

from elasticdl.proto import elasticdl_pb2_grpc
from elasticdl.python.common.args import (
    build_arguments_from_parsed_result,
    parse_envs,
)
from elasticdl.python.common.constants import (
    GRPC,
    JobType,
    WorkerManagerStatus,
)
from elasticdl.python.common.k8s_tensorboard_client import TensorBoardClient
from elasticdl.python.common.log_utils import get_logger
from elasticdl.python.common.model_utils import (
    find_layer,
    get_module_file_path,
    load_model_from_module,
    load_module,
)
from elasticdl.python.data.data_reader import create_data_reader
from elasticdl.python.elasticdl.layers.embedding import Embedding
from elasticdl.python.master.checkpoint_service import CheckpointService
from elasticdl.python.master.embedding_service import EmbeddingService
from elasticdl.python.master.evaluation_service import EvaluationService
from elasticdl.python.master.k8s_worker_manager import WorkerManager
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.master.task_dispatcher import _TaskDispatcher
from elasticdl.python.master.tensorboard_service import TensorboardService


def _make_task_dispatcher(
    training_data,
    evaluation_data,
    prediction_data,
    records_per_task,
    num_epochs,
):
    # TODO: Support any subclasses of `AbstractDataReader`
    # and support passing specified parameters to the constructor
    def _maybe_create_shards(data_origin):
        return (
            create_data_reader(
                data_origin=data_origin, records_per_task=records_per_task
            ).create_shards()
            if data_origin
            else {}
        )

    prediction_f_records = _maybe_create_shards(prediction_data)

    return _TaskDispatcher(
        _maybe_create_shards(training_data),
        _maybe_create_shards(evaluation_data),
        prediction_f_records,
        records_per_task,
        # Only generate prediction tasks for 1 epoch
        1 if prediction_f_records else num_epochs,
    )


class Master(object):
    logger = get_logger("master")

    def __init__(self, args):
        self.output = args.output

        # Master addr
        master_ip = os.getenv("MY_POD_IP", "localhost")
        self.master_addr = "%s:%d" % (master_ip, args.port)
        self.job_type = _get_job_type(args)

        # Start TensorBoard service if requested
        self.tb_service = self._create_tensorboard_service(
            args.tensorboard_log_dir, master_ip
        )

        # Start task queue
        records_per_task = args.minibatch_size * args.num_minibatches_per_task
        self.task_d = _make_task_dispatcher(
            args.training_data,
            args.evaluation_data,
            args.prediction_data,
            records_per_task,
            args.num_epochs,
        )

        # Initialize the components from the model definition
        self.model_module = load_module(
            get_module_file_path(args.model_zoo, args.model_def)
        ).__dict__
        self.model_inst = load_model_from_module(
            args.model_def, self.model_module, args.model_params
        )
        self.optimizer = self.model_module[args.optimizer]()

        # Initialize checkpoint service
        self.checkpoint_service = self._create_checkpoint_service(args) 

        # Initialize evaluation service
        self.evaluation_service = self._create_evaluation_service(args)

        # Search for embedding layers in the model,
        # if found, initialize embedding service
        layers = find_layer(self.model_inst, Embedding)
        self.embedding_service_endpoint, self.embedding_dims = self._create_embedding_service(
            layers, args
        )

        self.master_servicer, self.server = self._create_master_service(args)

        self.worker_manager = self._create_worker_manager(args)

    def start(self, args):
        # Start the evaluation service if requested
        if self.evaluation_service:
            logger.info("Starting evaluation service")
            self.evaluation_service.start()
            logger.info("Evaluation service started")

        # Start the master GRPC server
        logger.info("Starting master service")
        self.server.start()
        logger.info("Master service started")

        # Start the worker manager if requested
        if self.worker_manager:
            worker_manager.update_status(WorkerManagerStatus.PENDING)
            logger.info("Launching %d workers", args.num_workers)
            worker_manager.start_workers()
            worker_manager.update_status(WorkerManagerStatus.RUNNING)

        # Start TensorBoard k8s Service if requested
        if self.tb_service:
            logger.info("Starting tensorboard service")
            tb_service.start()
            TensorBoardClient(
                job_name=args.job_name,
                image_name=args.worker_image,
                namespace=args.namespace,
            ).start_tensorboard_service()
            logger.info("Tensorboard service started")

    def run(self):
        try:
            while True:
                if self.task_d.finished():
                    if self.worker_manager:
                        self.worker_manager.update_status(
                            WorkerManagerStatus.FINISHED
                        )
                    if self.output:
                        self.master_servicer.save_latest_checkpoint(
                            self.output
                        )
                    break
                time.sleep(30)
        except KeyboardInterrupt:
            logger.warning("Server stopping")

        self._cleanup()

    @staticmethod
    def _get_job_type(args):
        if all(
            (
                args.training_data,
                args.evaluation_data,
                args.evaluation_throttle_secs or args.evaluation_steps,
            )
        ):
            job_type = JobType.TRAINING_WITH_EVALUATION
        elif all(
            (
                args.evaluation_data,
                not args.training_data,
                not args.prediction_data,
            )
        ):
            job_type = JobType.EVALUATION_ONLY
        elif all(
            (
                args.prediction_data,
                not args.evaluation_data,
                not args.training_data,
            )
        ):
            job_type = JobType.PREDICTION_ONLY
        else:
            job_type = JobType.TRAINING_ONLY

        return job_type

    def _create_tensorboard_service(self, tensorboard_log_dir, master_ip):
        tb_service = None
        if tensorboard_log_dir:
            logger.info(
                "Create TensorBoard service with log directory %s",
                tensorboard_log_dir,
            )
            # Start TensorBoard CLI
            tb_service = TensorboardService(tensorboard_log_dir, master_ip)

        return tb_service

    def _create_checkpoint_service(self, args):
        checkpoint_service = None
        if (
            args.checkpoint_steps
            or self.job_type == JobType.TRAINING_WITH_EVALUATION
        ):
            logger.info("Create checkpoint service")
            checkpoint_service = CheckpointService(
                args.checkpoint_dir,
                args.checkpoint_steps,
                args.keep_checkpoint_max,
                self.job_type == JobType.TRAINING_WITH_EVALUATION,
            )

        return checkpoint_service

    def _create_evaluation_service(self, args):
        evaluation_service = None
        if (
            self.job_type == JobType.TRAINING_WITH_EVALUATION
            or self.job_type == JobType.EVALUATION_ONLY
        ):
            logger.info(
                "Create evaluation service with throttle seconds %d "
                " and evaluation steps %d",
                args.evaluation_throttle_secs,
                args.evaluation_steps,
            )
            evaluation_service = EvaluationService(
                self.checkpoint_service,
                self.tb_service,
                self.task_d,
                args.evaluation_start_delay_secs,
                args.evaluation_throttle_secs,
                args.evaluation_steps,
                self.job_type == JobType.EVALUATION_ONLY,
                self.model_module[args.eval_metrics_fn],
            )
            self.task_d.set_evaluation_service(evaluation_service)

        return evaluation_service

    def _create_embedding_service(self, layers, args):
        embedding_service_endpoint = None
        embedding_dims = {}

        if layers:
            embedding_service = EmbeddingService()
            embedding_service_endpoint = embedding_service.start_embedding_service(
                job_name=args.job_name,
                image_name=args.worker_image,
                namespace=args.namespace,
                resource_request=args.master_resource_request,
                resource_limit=args.master_resource_limit,
                pod_priority=args.worker_pod_priority,
                volume=args.volume,
                image_pull_policy=args.image_pull_policy,
                restart_policy=args.restart_policy,
                cluster_spec=args.cluster_spec,
            )
            logger.info(
                "Embedding service start succeeded. The endpoint is %s."
                % str(embedding_service_endpoint)
            )
            embedding_dims = dict(
                [(layer.name, layer.output_dim) for layer in layers]
            )

        return embedding_service_endpoint, embedding_dims

    def _create_master_service(self, args):
        # The master service
        logger.info("Create master service")
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
        master_servicer = MasterServicer(
            args.grads_to_wait,
            args.minibatch_size,
            self.optimizer,
            self.task_d,
            init_var=self.model_inst.trainable_variables
            if self.model_inst.built
            else [],
            embedding_dims=self.embedding_dims,
            checkpoint_filename_for_init=args.checkpoint_filename_for_init,
            checkpoint_service=self.checkpoint_service,
            evaluation_service=self.evaluation_service,
            embedding_service_endpoint=self.embedding_service_endpoint,
            lr_staleness_modulation=args.lr_staleness_modulation,
            use_async=args.use_async,
        )
        elasticdl_pb2_grpc.add_MasterServicer_to_server(
            master_servicer, server
        )
        server.add_insecure_port("[::]:{}".format(args.port))
        logger.info("The port of the master server is: %d", args.port)

        return master_servicer, server

    def _create_worker_manager(self, args):
        worker_manager = None
        if args.num_workers:
            assert args.worker_image, "Worker image cannot be empty"

            worker_command = ["python"]
            worker_args = [
                "-m",
                "elasticdl.python.worker.main",
                "--master_addr",
                self.master_addr,
                "--job_type",
                self.job_type,
                "--embedding_service_endpoint",
                str(self.embedding_service_endpoint),
            ]
            worker_args.extend(build_arguments_from_parsed_result(args))

            env_dict = parse_envs(args.envs)
            env = []
            for key in env_dict:
                env.append(V1EnvVar(name=key, value=env_dict[key]))

            worker_manager = WorkerManager(
                self.task_d,
                job_name=args.job_name,
                image_name=args.worker_image,
                command=worker_command,
                args=worker_args,
                namespace=args.namespace,
                num_workers=args.num_workers,
                worker_resource_request=args.worker_resource_request,
                worker_resource_limit=args.worker_resource_limit,
                pod_priority=args.worker_pod_priority,
                volume=args.volume,
                image_pull_policy=args.image_pull_policy,
                restart_policy=args.restart_policy,
                cluster_spec=args.cluster_spec,
                envs=env,
            )

        return worker_manager

    def _cleanup(self):
        if self.evaluation_service:
            logger.info("Stopping evaluation service")
            self.evaluation_service.stop()

        logger.info("Stopping RPC server")
        self.server.stop(0)

        # Keep TensorBoard running when all the tasks are finished
        if self.tb_service:
            logger.info(
                "All tasks finished. Keeping TensorBoard service running..."
            )
            while True:
                if self.tb_service.is_active():
                    time.sleep(10)
                else:
                    logger.warning(
                        "Unable to keep TensorBoard running. "
                        "It has already terminated"
                    )
                    break
        logger.info("Master stopped")
