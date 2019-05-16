import logging
import time
import argparse
import os

import grpc
import tensorflow as tf

tf.enable_eager_execution()

from concurrent import futures
from recordio import File
from elasticdl.proto import master_pb2_grpc
from elasticdl.master.servicer import MasterServicer
from elasticdl.master.task_queue import _TaskQueue
from elasticdl.master.k8s_worker_manager import WorkerManager
from elasticdl.common.model_helper import load_user_model, build_model


def _make_task_queue(data_dir, record_per_task, num_epoch):
    f_records = {}
    for f in os.listdir(data_dir):
        p = os.path.join(data_dir, f)
        with File(p, "r") as rio:
            f_records[p] = rio.count()
    return _TaskQueue(f_records, record_per_task, num_epoch)


def _parse_args():
    parser = argparse.ArgumentParser(description="ElasticDL Master")
    parser.add_argument(
        "--model_file",
        help="Full file path of user defined neural model",
        required=True,
    )
    parser.add_argument(
        "--train_data_dir",
        help="Training data directory. Files should be in RecordIO format",
        required=True,
    )
    parser.add_argument("--record_per_task", type=int, required=True)
    parser.add_argument("--num_epoch", type=int, required=True)
    parser.add_argument(
        "--grads_to_wait",
        type=int,
        help="Number of gradients to wait before updating model",
        required=True,
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        help="Minibatch size used by workers to compute gradients",
        required=True,
    )
    parser.add_argument(
        "--num_worker",
        type=int,
        help="the number of workers used in training",
        default=0,
    )
    parser.add_argument(
        "--worker_image", help="docker image for worker", default=None
    )
    parser.add_argument("--job_name", help="job name", default="elastic-train")
    parser.add_argument(
        "--codec-type",
        default=None,
        help="Type of codec(tf_example or None)",
    )
    return parser.parse_args()


def main():
    # TODO: pass port via flags.
    PORT = 50001
    logger = logging.getLogger("master")
    args = _parse_args()
    task_q = _make_task_queue(
        args.train_data_dir, args.record_per_task, args.num_epoch
    )
    model_module = load_user_model(args.model_file)
    model_inst = model_module.model
    build_model(model_inst, model_module.feature_columns())
    optimizer = model_module.optimizer()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=64))
    master_pb2_grpc.add_MasterServicer_to_server(
        MasterServicer(
            logger,
            args.grads_to_wait,
            args.minibatch_size,
            optimizer,
            task_q,
            init_var=model_inst.trainable_variables,
        ),
        server,
    )
    server.add_insecure_port("[::]:{}".format(PORT))
    server.start()
    logger.warning("Server started at port: %d", PORT)

    if args.num_worker:
        master_addr = "%s:%d" % (os.getenv("MY_POD_IP", "localhost"), PORT)
        worker_command = ["python"]
        worker_args = [
            "-m",
            "elasticdl.worker.main",
            "--model_file",
            args.model_file,
            "--master_addr",
            master_addr,
            "--codec-type",
            args.codec_type
        ]

        worker_manager = WorkerManager(
            job_name=args.job_name,
            worker_image=args.worker_image,
            command=worker_command,
            args=worker_args,
            namespace="default",
            num_worker=args.num_worker,
        )
        worker_manager.start_workers(restart_policy="Never")

    try:
        while True:
            if task_q.finished():
                break
            time.sleep(30)
    except KeyboardInterrupt:
        logger.warning("Server stopping")

    if args.num_worker:
        # TODO: worker_manager.remove_workers supports synchronized call
        worker_manager.remove_workers()
        # wait for worker pod to be deleted
        max_check_num = 10
        for _ in range(max_check_num):
            time.sleep(3)
            counters = worker_manager.get_counters()
            if not counters:
                break
    server.stop(0)


if __name__ == "__main__":
    logging.basicConfig()
    main()
