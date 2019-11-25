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
    InstanceManagerStatus,
    JobType,
)
from elasticdl.python.common.k8s_tensorboard_client import TensorBoardClient
from elasticdl.python.common.log_utils import get_logger
from elasticdl.python.common.model_handler import ModelHandler
from elasticdl.python.common.model_utils import (
    get_module_file_path,
    load_model_from_module,
    load_module,
)
from elasticdl.python.data.data_reader import create_data_reader
from elasticdl.python.master.evaluation_service import EvaluationService
from elasticdl.python.master.k8s_instance_manager import InstanceManager
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.master.task_dispatcher import _TaskDispatcher
from elasticdl.python.master.tensorboard_service import TensorboardService


def _make_task_dispatcher(
    training_data,
    validation_data,
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
        _maybe_create_shards(validation_data),
        prediction_f_records,
        records_per_task,
        # Only generate prediction tasks for 1 epoch
        1 if prediction_f_records else num_epochs,
    )


class Master(object):
    def __init__(self, args):
        self.logger = get_logger("master", level=args.log_level.upper())

        self.num_ps_pods = args.num_ps_pods
        self.checkpoint_output_path = args.checkpoint_dir

        # Master addr
        master_ip = os.getenv("MY_POD_IP", "localhost")
        self.master_addr = "%s:%d" % (master_ip, args.port)
        self.job_type = Master._get_job_type(args)

        # Initialize TensorBoard service if requested
        self.tb_service = self._create_tensorboard_service(
            args.tensorboard_log_dir, master_ip
        )
        if self.tb_service:
            self.tb_client = TensorBoardClient(
                job_name=args.job_name,
                image_name=args.worker_image,
                namespace=args.namespace,
            )

        # Start task queue
        records_per_task = args.minibatch_size * args.num_minibatches_per_task
        self.task_d = _make_task_dispatcher(
            args.training_data,
            args.validation_data,
            args.prediction_data,
            records_per_task,
            args.num_epochs,
        )

        saved_model_path = args.output
        if saved_model_path is not None and self.job_type in [
            JobType.TRAINING_ONLY,
            JobType.TRAINING_WITH_EVALUATION,
        ]:
            self.task_d.add_deferred_callback_create_save_model_task(
                saved_model_path
            )

        # Initialize the components from the model definition
        self.model_module = load_module(
            get_module_file_path(args.model_zoo, args.model_def)
        ).__dict__
        self.model_inst = load_model_from_module(
            args.model_def, self.model_module, args.model_params
        )
        model_handler = ModelHandler.get_model_handler(
            args.distribution_strategy
        )
        self.model_inst = model_handler.get_model_to_train(self.model_inst)
        self.optimizer = self.model_module[args.optimizer]()

        self.evaluation_service = self._create_evaluation_service(args)

        # Initialize master service
        self.master_servicer, self.server = self._create_master_service(args)

        # Initialize instance manager
        self.instance_manager = self._create_instance_manager(args)

        self._should_stop = False
        self._exit_code = 0

    def request_stop(self, err_msg=None):
        """Request master to quit"""
        self._should_stop = True
        if err_msg:
            self.logger.error(err_msg)
            # TODO (chengfu.wcy) create meaningful status codes
            self._exit_code = -1

    def prepare(self):
        """
        Start the components one by one. Make sure that it is ready to run.
        """
        # Start the evaluation service if requested
        if self.evaluation_service:
            self.logger.info("Starting evaluation service")
            self.evaluation_service.start()
            self.logger.info("Evaluation service started")

        # Start the master GRPC server
        self.logger.info("Starting master RPC server")
        self.server.start()
        self.logger.info("Master RPC server started")

        # Start the worker manager if requested
        if self.instance_manager:
            self.instance_manager.update_status(InstanceManagerStatus.PENDING)
            self.instance_manager.start_all_ps()
            self.instance_manager.start_workers()
            self.instance_manager.update_status(InstanceManagerStatus.RUNNING)

        # Start TensorBoard k8s Service if requested
        if self.tb_service and self.tb_client:
            self.logger.info("Starting tensorboard service")
            self.tb_service.start()
            self.tb_client.start_tensorboard_service()
            self.logger.info("Tensorboard service started")

    def run(self):
        """
        The main loop of master.
        Dispatch the tasks to the workers until all the tasks are completed.
        """
        try:
            while True:
                if self.task_d.finished():
                    if self.instance_manager:
                        self.instance_manager.update_status(
                            InstanceManagerStatus.FINISHED
                        )
                    break
                if self._should_stop:
                    break
                time.sleep(30)
        except KeyboardInterrupt:
            self.logger.warning("Server stopping")
        finally:
            self._stop()
        return self._exit_code

    def _stop(self):
        """
        Stop all the components.
        Make sure that the created services and components are shut down.
        """
        self.logger.info("Stopping master")

        if self.evaluation_service:
            self.logger.info("Stopping evaluation service")
            self.evaluation_service.stop()
            self.logger.info("Evaluation service stopped")

        self.logger.info("Stopping RPC server")
        self.server.stop(None)  # grace = None
        self.logger.info("RPC server stopped")

        # Keep TensorBoard running when all the tasks are finished
        if self.tb_service:
            self.logger.info(
                "All tasks finished. Keeping TensorBoard service running..."
            )
            while True:
                if self.tb_service.is_active():
                    time.sleep(10)
                else:
                    self.logger.warning(
                        "Unable to keep TensorBoard running. "
                        "It has already terminated"
                    )
                    break
        self.logger.info("Master stopped")

    @staticmethod
    def _get_job_type(args):
        if all(
            (
                args.training_data,
                args.validation_data,
                args.evaluation_throttle_secs or args.evaluation_steps,
            )
        ):
            job_type = JobType.TRAINING_WITH_EVALUATION
        elif all(
            (
                args.validation_data,
                not args.training_data,
                not args.prediction_data,
            )
        ):
            job_type = JobType.EVALUATION_ONLY
        elif all(
            (
                args.prediction_data,
                not args.validation_data,
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
            self.logger.info(
                "Creating TensorBoard service with log directory %s",
                tensorboard_log_dir,
            )
            # Start TensorBoard CLI
            tb_service = TensorboardService(tensorboard_log_dir, master_ip)

        return tb_service

    def _create_evaluation_service(self, args):
        evaluation_service = None
        if (
            self.job_type == JobType.TRAINING_WITH_EVALUATION
            or self.job_type == JobType.EVALUATION_ONLY
        ):
            self.logger.info(
                "Creating evaluation service with throttle seconds %d "
                " and evaluation steps %d",
                args.evaluation_throttle_secs,
                args.evaluation_steps,
            )
            evaluation_service = EvaluationService(
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

    def _create_master_service(self, args):
        self.logger.info("Creating master service")
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
            args.minibatch_size,
            self.task_d,
            evaluation_service=self.evaluation_service,
        )
        elasticdl_pb2_grpc.add_MasterServicer_to_server(
            master_servicer, server
        )
        server.add_insecure_port("[::]:{}".format(args.port))
        self.logger.info("The port of the master server is: %d", args.port)

        return master_servicer, server

    def _create_instance_manager(self, args):
        instance_manager = None
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
            ]
            worker_args.extend(build_arguments_from_parsed_result(args))

            ps_command = ["python"]
            ps_args = [
                "-m",
                "elasticdl.python.ps.main",
                "--grads_to_wait",
                str(args.grads_to_wait),
                "--lr_staleness_modulation",
                str(args.lr_staleness_modulation),
                "--use_async",
                str(args.use_async),
                "--minibatch_size",
                str(args.minibatch_size),
                "--model_zoo",
                args.model_zoo,
                "--model_def",
                args.model_def,
                "--job_name",
                args.job_name,
                "--num_minibatches_per_task",
                str(args.num_minibatches_per_task),
                "--port",
                "2222",
                "--master_addr",
                self.master_addr,
                "--namespace",
                args.namespace,
                "--evaluation_steps",
                str(args.evaluation_steps),
                "--checkpoint_dir",
                str(args.checkpoint_dir),
                "--checkpoint_steps",
                str(args.checkpoint_steps),
                "--keep_checkpoint_max",
                str(args.keep_checkpoint_max),
                "--num_ps_pods",
                str(args.num_ps_pods),
            ]

            env_dict = parse_envs(args.envs)
            env = []
            for key in env_dict:
                env.append(V1EnvVar(name=key, value=env_dict[key]))

            instance_manager = InstanceManager(
                self.task_d,
                job_name=args.job_name,
                image_name=args.worker_image,
                worker_command=worker_command,
                worker_args=worker_args,
                namespace=args.namespace,
                num_workers=args.num_workers,
                worker_resource_request=args.worker_resource_request,
                worker_resource_limit=args.worker_resource_limit,
                worker_pod_priority=args.worker_pod_priority,
                num_ps=args.num_ps_pods,
                ps_command=ps_command,
                ps_args=ps_args,
                ps_resource_request=args.ps_resource_request,
                ps_resource_limit=args.ps_resource_limit,
                ps_pod_priority=args.ps_pod_priority,
                volume=args.volume,
                image_pull_policy=args.image_pull_policy,
                restart_policy=args.restart_policy,
                cluster_spec=args.cluster_spec,
                envs=env,
            )

        return instance_manager
