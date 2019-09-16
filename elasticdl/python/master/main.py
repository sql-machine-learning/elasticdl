import os
import time
from concurrent import futures
from contextlib import closing

import grpc
import recordio
from kubernetes.client import V1EnvVar

from elasticdl.proto import elasticdl_pb2_grpc
from elasticdl.python.common.args import parse_envs, parse_master_args
from elasticdl.python.common.constants import (
    GRPC,
    JobType,
    WorkerManagerStatus,
)
from elasticdl.python.common.k8s_tensorboard_client import TensorBoardClient
from elasticdl.python.common.log_util import get_logger
from elasticdl.python.common.model_helper import (
    find_layer,
    get_module_file_path,
    load_model_from_module,
    load_module,
)
from elasticdl.python.elasticdl.layers.embedding import Embedding
from elasticdl.python.master.checkpoint_service import CheckpointService
from elasticdl.python.master.embedding_service import EmbeddingService
from elasticdl.python.master.evaluation_service import EvaluationService
from elasticdl.python.master.k8s_worker_manager import WorkerManager
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.master.task_dispatcher import _TaskDispatcher
from elasticdl.python.master.tensorboard_service import TensorboardService


def _make_task_dispatcher(
    training_data_dir,
    evaluation_data_dir,
    prediction_data_dir,
    records_per_task,
    num_epochs,
):
    def _collect_file_records_from_dir(data_dir):
        if not data_dir:
            return {}
        f_records = {}
        for f in os.listdir(data_dir):
            p = os.path.join(data_dir, f)
            with closing(recordio.Index(p)) as rio:
                f_records[p] = rio.num_records()
        return f_records

    training_f_records = _collect_file_records_from_dir(training_data_dir)
    evaluation_f_records = _collect_file_records_from_dir(evaluation_data_dir)
    prediction_f_records = _collect_file_records_from_dir(prediction_data_dir)

    return _TaskDispatcher(
        training_f_records,
        evaluation_f_records,
        prediction_f_records,
        records_per_task,
        # Only generate prediction tasks for 1 epoch
        1 if prediction_f_records else num_epochs,
    )


def main():
    args = parse_master_args()
    logger = get_logger("master", level=args.log_level.upper())

    # Master addr
    master_ip = os.getenv("MY_POD_IP", "localhost")
    master_addr = "%s:%d" % (master_ip, args.port)

    # Start TensorBoard service if requested
    if args.tensorboard_log_dir:
        logger.info(
            "Starting TensorBoard service with log directory %s",
            args.tensorboard_log_dir,
        )
        # Start TensorBoard CLI
        tb_service = TensorboardService(args.tensorboard_log_dir, master_ip)
        tb_service.start()
    else:
        tb_service = None

    # Start task queue
    logger.debug(
        "Starting task queue with training data directory %s, "
        "evaluation data directory %s, "
        "and prediction data directory %s",
        args.training_data_dir,
        args.evaluation_data_dir,
        args.prediction_data_dir,
    )
    task_d = _make_task_dispatcher(
        args.training_data_dir,
        args.evaluation_data_dir,
        args.prediction_data_dir,
        args.records_per_task,
        args.num_epochs,
    )
    model_module = load_module(
        get_module_file_path(args.model_zoo, args.model_def)
    ).__dict__
    model_inst = load_model_from_module(
        args.model_def, model_module, args.model_params
    )
    optimizer = model_module[args.optimizer]()

    if all(
        (
            args.training_data_dir,
            args.evaluation_data_dir,
            args.evaluation_throttle_secs or args.evaluation_steps,
        )
    ):
        job_type = JobType.TRAINING_WITH_EVALUATION
    elif all(
        (
            args.evaluation_data_dir,
            not args.training_data_dir,
            not args.prediction_data_dir,
        )
    ):
        job_type = JobType.EVALUATION_ONLY
    elif all(
        (
            args.prediction_data_dir,
            not args.evaluation_data_dir,
            not args.training_data_dir,
        )
    ):
        job_type = JobType.PREDICTION_ONLY
    else:
        job_type = JobType.TRAINING_ONLY

    # Initialize checkpoint service
    if args.checkpoint_steps or job_type == JobType.TRAINING_WITH_EVALUATION:
        logger.info("Starting checkpoint service")
        checkpoint_service = CheckpointService(
            args.checkpoint_dir,
            args.checkpoint_steps,
            args.keep_checkpoint_max,
            job_type == JobType.TRAINING_WITH_EVALUATION,
        )
    else:
        checkpoint_service = None

    # Initialize evaluation service
    evaluation_service = None
    if (
        job_type == JobType.TRAINING_WITH_EVALUATION
        or job_type == JobType.EVALUATION_ONLY
    ):
        logger.info(
            "Starting evaluation service with throttle seconds %d "
            " and evaluation steps %d",
            args.evaluation_throttle_secs,
            args.evaluation_steps,
        )
        evaluation_service = EvaluationService(
            checkpoint_service,
            tb_service,
            task_d,
            args.evaluation_start_delay_secs,
            args.evaluation_throttle_secs,
            args.evaluation_steps,
            job_type == JobType.EVALUATION_ONLY,
        )
        evaluation_service.start()
        task_d.set_evaluation_service(evaluation_service)

    embedding_service_endpoint = None
    embedding_dims = {}
    # Search for embedding layers in the model,
    # if found, initialize embedding service
    layers = find_layer(model_inst, Embedding)
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

    # The master service
    logger.info("Starting master service")
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
        optimizer,
        task_d,
        init_var=model_inst.trainable_variables if model_inst.built else [],
        embedding_dims=embedding_dims,
        checkpoint_filename_for_init=args.checkpoint_filename_for_init,
        checkpoint_service=checkpoint_service,
        evaluation_service=evaluation_service,
        embedding_service_endpoint=embedding_service_endpoint,
        lr_staleness_modulation=args.lr_staleness_modulation,
        use_async=args.use_async,
    )
    elasticdl_pb2_grpc.add_MasterServicer_to_server(master_servicer, server)
    server.add_insecure_port("[::]:{}".format(args.port))
    server.start()
    logger.info("Server started at port: %d", args.port)

    worker_manager = None
    if args.num_workers:
        assert args.worker_image, "Worker image cannot be empty"

        worker_command = ["python"]
        worker_args = [
            "-m",
            "elasticdl.python.worker.main",
            "--model_zoo",
            args.model_zoo,
            "--master_addr",
            master_addr,
            "--log_level",
            args.log_level,
            "--dataset_fn",
            args.dataset_fn,
            "--loss",
            args.loss,
            "--optimizer",
            args.optimizer,
            "--eval_metrics_fn",
            args.eval_metrics_fn,
            "--model_def",
            args.model_def,
            "--job_type",
            job_type,
            "--minibatch_size",
            str(args.minibatch_size),
            "--embedding_service_endpoint",
            str(embedding_service_endpoint),
            "--get_model_steps",
            str(args.get_model_steps),
        ]

        env_dict = parse_envs(args.envs)
        env = []
        for key in env_dict:
            env.append(V1EnvVar(name=key, value=env_dict[key]))

        worker_manager = WorkerManager(
            task_d,
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
        worker_manager.update_status(WorkerManagerStatus.PENDING)
        logger.info("Launching %d workers", args.num_workers)
        worker_manager.start_workers()
        worker_manager.update_status(WorkerManagerStatus.RUNNING)

    # Start TensorBoard k8s Service if requested
    if tb_service:
        TensorBoardClient(
            job_name=args.job_name,
            image_name=args.worker_image,
            namespace=args.namespace,
        ).start_tensorboard_service()

    try:
        while True:
            if task_d.finished():
                if worker_manager:
                    worker_manager.update_status(WorkerManagerStatus.FINISHED)
                if args.output:
                    master_servicer.save_latest_checkpoint(args.output)
                break
            time.sleep(30)
    except KeyboardInterrupt:
        logger.warning("Server stopping")

    if evaluation_service:
        logger.info("Stopping evaluation service")
        evaluation_service.stop()

    logger.info("Stopping RPC server")
    server.stop(0)

    # Keep TensorBoard running when all the tasks are finished
    if tb_service:
        logger.info(
            "All tasks finished. Keeping TensorBoard service running..."
        )
        while True:
            if tb_service.is_active():
                time.sleep(10)
            else:
                logger.warning(
                    "Unable to keep TensorBoard running. "
                    "It has already terminated"
                )
                break
    logger.info("Master stopped")


if __name__ == "__main__":
    main()
