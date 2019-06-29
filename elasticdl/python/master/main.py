import logging
import time
import os
import recordio

import grpc

from contextlib import closing
from concurrent import futures
from elasticdl.proto import elasticdl_pb2_grpc
from elasticdl.python.master.args import parse_args
from elasticdl.python.master.checkpoint_service import CheckpointService
from elasticdl.python.master.evaluation_service import EvaluationService
from elasticdl.python.master.tensorboard_service import TensorboardService
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.master.task_queue import _TaskQueue
from elasticdl.python.master.k8s_worker_manager import WorkerManager
from elasticdl.python.common.model_helper import load_module
from elasticdl.python.common.constants import GRPC


def _make_task_queue(
    training_data_dir, evaluation_data_dir, records_per_task, num_epochs
):
    def _collect_file_records_from_dir(data_dir):
        f_records = {}
        for f in os.listdir(data_dir):
            p = os.path.join(data_dir, f)
            with closing(recordio.Index(p)) as rio:
                f_records[p] = rio.num_records()
        return f_records

    training_f_records = _collect_file_records_from_dir(training_data_dir)
    evaluation_f_records = (
        {}
        if evaluation_data_dir == ""
        else _collect_file_records_from_dir(evaluation_data_dir)
    )
    return _TaskQueue(
        training_f_records, evaluation_f_records, records_per_task, num_epochs
    )


def main():
    args = parse_args()

    # Initialize logger and set level for ROOT logger
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)-8s "
        "[%(filename)s:%(lineno)d] %(message)s"
    )
    logging.getLogger().setLevel(args.log_level)
    logger = logging.getLogger(__name__)

    # Start tensorboard service if required
    if args.tensorboard_log_dir:
        logger.info(
            "Starting tensorboard service with log directory %s",
            args.tensorboard_log_dir,
        )
        tb_service = TensorboardService(args.tensorboard_log_dir)
        tb_service.start()
    else:
        tb_service = None

    # Start task queue
    logger.info(
        "Starting task queue with training data directory %s "
        "and evaluation data directory %s",
        args.training_data_dir,
        args.evaluation_data_dir,
    )
    task_q = _make_task_queue(
        args.training_data_dir,
        args.evaluation_data_dir,
        args.records_per_task,
        args.num_epochs,
    )
    model_module = load_module(args.model_file)
    model_inst = model_module.model
    optimizer = model_module.optimizer()

    # Initialize checkpoint service
    if args.checkpoint_steps:
        logger.info("Starting checkpoint service")
        checkpoint_service = CheckpointService(
            args.checkpoint_dir,
            args.checkpoint_steps,
            args.keep_checkpoint_max,
        )
    else:
        checkpoint_service = None

    # Initialize evaluation service
    evaluation_service = None
    if args.evaluation_data_dir:
        if args.checkpoint_steps <= 0:
            raise ValueError(
                "Checkpoint should also be enabled when evaluation is enabled"
            )
        logger.info(
            "Starting evaluation service with throttle seconds %d",
            args.evaluation_throttle_secs,
        )
        evaluation_service = EvaluationService(
            checkpoint_service,
            tb_service,
            task_q,
            args.evaluation_start_delay_secs,
            args.evaluation_throttle_secs,
        )
        evaluation_service.start()
        task_q.set_evaluation_service(evaluation_service)

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
        task_q,
        init_var=model_inst.trainable_variables if model_inst.built else [],
        checkpoint_filename_for_init=args.checkpoint_filename_for_init,
        checkpoint_service=checkpoint_service,
        evaluation_service=evaluation_service,
    )
    elasticdl_pb2_grpc.add_MasterServicer_to_server(master_servicer, server)
    server.add_insecure_port("[::]:{}".format(args.port))
    server.start()
    logger.info("Server started at port: %d", args.port)

    if args.num_workers:
        assert args.worker_image, "Worker image cannot be empty"

        master_addr = "%s:%d" % (
            os.getenv("MY_POD_IP", "localhost"),
            args.port,
        )
        worker_command = ["python"]
        worker_args = [
            "-m",
            "elasticdl.python.worker.main",
            "--model_file",
            args.model_file,
            "--master_addr",
            master_addr,
            "--log_level",
            args.log_level,
        ]

        args.worker_resource_limit = (
            args.worker_resource_limit
            if args.worker_resource_limit
            else args.worker_resource_request
        )

        worker_manager = WorkerManager(
            task_q,
            job_name=args.job_name,
            image_name=args.worker_image,
            command=worker_command,
            args=worker_args,
            namespace=args.namespace,
            num_workers=args.num_workers,
            worker_resource_request=args.worker_resource_request,
            worker_resource_limit=args.worker_resource_limit,
            pod_priority=args.worker_pod_priority,
            mount_path=args.mount_path,
            volume_name=args.volume_name,
            image_pull_policy=args.image_pull_policy,
            restart_policy=args.restart_policy,
        )
        logger.info("Launching %d workers", args.num_workers)
        worker_manager.start_workers()

        if tb_service:
            worker_manager.start_tensorboard_service()

    try:
        while True:
            if task_q.finished():
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
        tb_service.keep_running()
    logger.info("Master stopped")


if __name__ == "__main__":
    main()
