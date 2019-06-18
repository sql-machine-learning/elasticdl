import logging
import time
import argparse
import os
import recordio

import grpc

from contextlib import closing
from concurrent import futures
from threading import Event
from elasticdl.proto import elasticdl_pb2_grpc
from elasticdl.python.elasticdl.master.checkpoint_service import (
    CheckpointService,
)
from elasticdl.python.elasticdl.master.servicer import MasterServicer
from elasticdl.python.elasticdl.master.task_queue import (
    _EvaluationTrigger,
    _TaskQueue,
)
from elasticdl.python.elasticdl.master.k8s_worker_manager import WorkerManager
from elasticdl.python.elasticdl.common.model_helper import load_user_model


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


def _pos_int(arg):
    res = int(arg)
    if res <= 0:
        raise ValueError("Positive integer argument required. Got %s" % res)
    return res


def _non_neg_int(arg):
    res = int(arg)
    if res < 0:
        raise ValueError(
            "Non-negative integer argument required. Get %s" % res
        )
    return res


def _parse_args():
    parser = argparse.ArgumentParser(description="ElasticDL Master")
    parser.add_argument(
        "--model_file",
        help="Full file path of user defined neural model",
        required=True,
    )
    parser.add_argument(
        "--training_data_dir",
        help="Training data directory. Files should be in RecordIO format",
        required=True,
    )
    parser.add_argument(
        "--evaluation_data_dir",
        help="Evaluation data directory. Files should be in RecordIO format",
        default="",
    )
    parser.add_argument(
        "--evaluation_start_delay_secs",
        type=_pos_int,
        help="Start evaluation only after waiting for this many seconds",
        default=100,
    )
    parser.add_argument(
        "--evaluation_throttle_secs",
        type=_pos_int,
        help="Do not re-evaluate unless the last evaluation was started "
        "at least this many seconds ago",
        default=100,
    )
    parser.add_argument("--records_per_task", type=_pos_int, required=True)
    parser.add_argument("--num_epochs", type=_pos_int, required=True)
    parser.add_argument(
        "--grads_to_wait",
        type=_pos_int,
        help="Number of gradients to wait before updating model",
        required=True,
    )
    parser.add_argument(
        "--minibatch_size",
        type=_pos_int,
        help="Minibatch size used by workers to compute gradients",
        required=True,
    )
    parser.add_argument(
        "--num_workers", type=_pos_int, help="Number of workers", default=0
    )
    parser.add_argument(
        "--init_from_checkpoint",
        help="The checkpoint file to initialize the training model",
        default="",
    )
    parser.add_argument(
        "--checkpoint_dir",
        help="The directory to store the checkpoint files",
        default="",
    )
    parser.add_argument(
        "--checkpoint_steps",
        type=_non_neg_int,
        help="Save checkpoint every this many steps."
        "If 0, no checkpoints to save.",
        default=0,
    )
    parser.add_argument(
        "--keep_checkpoint_max",
        type=_non_neg_int,
        help="The maximum number of recent checkpoint files to keep."
        "If 0, keep all.",
        default=3,
    )
    parser.add_argument(
        "--worker_resource_request",
        default="cpu=1,memory=4096Mi",
        type=str,
        help="The minimal resource required by worker, "
        "e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1",
    )
    parser.add_argument(
        "--worker_resource_limit",
        default="cpu=1,memory=4096Mi",
        type=str,
        help="The maximal resource required by worker, "
        "e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1",
    )
    parser.add_argument(
        "--worker_pod_priority", help="Priority requested by workers"
    )
    parser.add_argument(
        "--worker_image", help="Docker image for workers", default=None
    )
    parser.add_argument("--job_name", help="Job name", required=True)
    parser.add_argument(
        "--codec_type",
        default="bytes",
        choices=["tf_example", "bytes"],
        help="Type of codec (tf_example or bytes)",
    )
    # TODO: better logic for handling volume configs
    parser.add_argument(
        "--volume_name", help="Volume name of Network File System"
    )
    parser.add_argument(
        "--mount_path", help="Mount path in the docker container"
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        default="WARNING",
        help="The logging level. Default to WARNING",
    )
    parser.add_argument(
        "--image_pull_policy", help="Image pull policy of master and workers"
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # TODO: pass port via flags.
    PORT = 50001

    # Initialize logger
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)-8s "
        "[%(filename)s:%(lineno)d] %(message)s"
    )
    # Set level for ROOT logger.
    logging.getLogger().setLevel(args.log_level)
    logger = logging.getLogger(__name__)

    task_q = _make_task_queue(
        args.training_data_dir,
        args.evaluation_data_dir,
        args.records_per_task,
        args.num_epochs,
    )

    model_module = load_user_model(args.model_file)
    model_inst = model_module.model
    optimizer = model_module.optimizer()

    # Initialize checkpoint service
    checkpoint_service = CheckpointService(
        args.checkpoint_dir, args.checkpoint_steps, args.keep_checkpoint_max
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=64))
    master_servicer = MasterServicer(
        args.grads_to_wait,
        args.minibatch_size,
        optimizer,
        task_q,
        init_var=model_inst.trainable_variables if model_inst.built else [],
        init_from_checkpoint=args.init_from_checkpoint,
        checkpoint_service=checkpoint_service,
    )
    elasticdl_pb2_grpc.add_MasterServicer_to_server(master_servicer, server)
    server.add_insecure_port("[::]:{}".format(PORT))
    server.start()
    logger.info("Server started at port: %d", PORT)

    stop_flag = Event()
    if args.evaluation_data_dir:
        if args.checkpoint_steps <= 0:
            raise ValueError(
                "Checkpoint should also be enabled when evaluation is enabled"
            )
        evaluation_timer = _EvaluationTrigger(
            master_servicer,
            task_q,
            stop_flag,
            args.evaluation_start_delay_secs,
            args.evaluation_throttle_secs,
        )
        evaluation_timer.start()

    if args.num_workers:
        assert args.worker_image, "Worker image cannot be empty"

        master_addr = "%s:%d" % (os.getenv("MY_POD_IP", "localhost"), PORT)
        worker_command = ["python"]
        worker_args = [
            "-m",
            "elasticdl.python.elasticdl.worker.main",
            "--model_file",
            args.model_file,
            "--master_addr",
            master_addr,
            "--codec_type",
            args.codec_type,
            "--log_level",
            args.log_level,
        ]

        worker_manager = WorkerManager(
            task_q,
            job_name=args.job_name,
            worker_image=args.worker_image,
            command=worker_command,
            args=worker_args,
            namespace="default",
            num_workers=args.num_workers,
            worker_resource_request=args.worker_resource_request,
            worker_resource_limit=args.worker_resource_limit,
            pod_priority=args.worker_pod_priority,
            mount_path=args.mount_path,
            volume_name=args.volume_name,
            image_pull_policy=args.image_pull_policy,
            restart_policy="Never",
        )
        worker_manager.start_workers()

    try:
        while True:
            if task_q.finished():
                stop_flag.set()
                break
            time.sleep(30)
    except KeyboardInterrupt:
        stop_flag.set()
        logger.warning("Server stopping")

    server.stop(0)


if __name__ == "__main__":
    main()
