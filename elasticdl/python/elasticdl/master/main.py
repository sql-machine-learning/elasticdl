import logging
import time
import argparse
import os
import recordio

import grpc
import tensorflow as tf

from contextlib import closing
from concurrent import futures
from elasticdl.proto import elasticdl_pb2_grpc
from elasticdl.python.elasticdl.master.servicer import MasterServicer
from elasticdl.python.elasticdl.master.task_queue import _TaskQueue
from elasticdl.python.elasticdl.master.k8s_worker_manager import WorkerManager
from elasticdl.python.elasticdl.common.model_helper import (
    load_user_model,
    build_model,
)

tf.enable_eager_execution()


def _make_task_queue(training_data_dir, evaluation_data_dir,
                     records_per_task, num_epochs):
    def _collect_file_records_from_dir(data_dir):
        f_records = {}
        for f in os.listdir(data_dir):
            p = os.path.join(data_dir, f)
            with closing(recordio.Index(p)) as rio:
                f_records[p] = rio.num_records()
        return f_records
    training_f_records = _collect_file_records_from_dir(training_data_dir)
    evaluation_f_records = (
        {} if evaluation_data_dir == ""
        else _collect_file_records_from_dir(evaluation_data_dir))
    return _TaskQueue(training_f_records, evaluation_f_records,
                      records_per_task, num_epochs)


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
        "--records_per_task",
        type=_pos_int,
        required=True,
    )
    parser.add_argument(
        "--num_epochs",
        type=_pos_int,
        required=True,
    )
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
        "--num_workers",
        type=_pos_int,
        help="Number of workers",
        default=0,
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
        help="Save checkpoint every this many steps. \
              If 0, no checkpoints to save.",
        default=0,
    )
    parser.add_argument(
        "--keep_checkpoint_max",
        type=_non_neg_int,
        help="The maximum number of recent checkpoint files to keep. \
              If 0, keep all.",
        default=3,
    )
    parser.add_argument(
        "--worker_cpu_request",
        help="The minimal CPU required by each worker",
        default="1000m",
    )
    parser.add_argument(
        "--worker_cpu_limit",
        help="The maximal CPU used by each worker",
        default="1000m",
    )
    parser.add_argument(
        "--worker_memory_request",
        help="The minimal memory required by each worker",
        default="4096Mi",
    )
    parser.add_argument(
        "--worker_memory_limit",
        help="The maximal memory used by each worker",
        default="4096Mi",
    )
    parser.add_argument(
        "--worker_pod_priority",
        help="Priority requested by workers",
    )
    parser.add_argument(
        "--worker_image",
        help="Docker image for workers",
        default=None,
    )
    parser.add_argument(
        "--job_name",
        help="Job name",
        required=True
    )
    parser.add_argument(
        "--codec_type",
        default="bytes",
        choices=["tf_example", "bytes"],
        help="Type of codec (tf_example or bytes)",
    )
    # TODO: better logic for handling volume configs
    parser.add_argument(
        "--volume_name",
        help="Volume name of Network File System"
    )
    parser.add_argument(
        "--mount_path",
        help="Mount path in the docker container",
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        default="WARNING",
        help="The logging level. Default to WARNING",
    )
    parser.add_argument(
        "--image_pull_policy",
        help="Image pull policy of master and workers",
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
        args.num_epochs
    )
    model_module = load_user_model(args.model_file)
    model_inst = model_module.model
    build_model(model_inst, model_module.feature_columns())
    optimizer = model_module.optimizer()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=64))
    elasticdl_pb2_grpc.add_MasterServicer_to_server(
        MasterServicer(
            args.grads_to_wait,
            args.minibatch_size,
            optimizer,
            task_q,
            init_var=model_inst.trainable_variables,
            init_from_checkpoint=args.init_from_checkpoint,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_steps=args.checkpoint_steps,
            keep_checkpoint_max=args.keep_checkpoint_max,
        ),
        server,
    )
    server.add_insecure_port("[::]:{}".format(PORT))
    server.start()
    logger.info("Server started at port: %d", PORT)

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
            cpu_request=args.worker_cpu_request,
            cpu_limit=args.worker_cpu_limit,
            memory_request=args.worker_memory_request,
            memory_limit=args.worker_memory_limit,
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
                break
            time.sleep(30)
    except KeyboardInterrupt:
        logger.warning("Server stopping")

    server.stop(0)


if __name__ == "__main__":
    main()
