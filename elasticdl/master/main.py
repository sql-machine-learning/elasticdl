from concurrent import futures
import logging
import time
import argparse
import os

import grpc
import tensorflow as tf

tf.enable_eager_execution()

from recordio import File
from elasticdl.proto import master_pb2_grpc
from elasticdl.master.servicer import MasterServicer
from elasticdl.master.task_queue import _TaskQueue
from elasticdl.master.k8s_worker_servicer import WorkerServicer
from elasticdl.common.model_helper import load_user_model


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
        "--model-file",
        help="Full file path of user defined neural model",
        required=True,
    )
    parser.add_argument(
        "--model-class",
        help="The model class name defined in model file",
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
        "--worker_image",
        help="docker image for worker",
        default=None,
    )
    parser.add_argument(
        "--job_name",
        help="job name",
        default="elastic-train",
    )
    return parser.parse_args()


def main():
    logger = logging.getLogger("master")
    args = _parse_args()
    task_q = _make_task_queue(
        args.train_data_dir, args.record_per_task, args.num_epoch
    )
    model_cls = load_user_model(args.model_file, args.model_class)
    model_inst = model_cls()
    model_inst.build(model_inst.input_shapes())
    optimizer = model_cls.optimizer()

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
    server.add_insecure_port("[::]:50001")
    server.start()
    logger.warning("Server started")

    if args.num_worker:
        # get master pod IP from env
        pod_ip = os.getenv("MY_POD_IP", "localhost")
        master_addr = pod_ip + ":50001"
        worker_command = ["python"]
        worker_args = [
                "-m",
                "elasticdl.worker.main",
                "--model-file={}".format(args.model_file),
                "--model-class={}".format(args.model_class),
                "--master_addr={}".format(master_addr)
            ]

        worker_servicer = WorkerServicer(
                job_name=args.job_name,
                worker_image=args.worker_image,
                command=worker_command,
                args=worker_args,
                namespace="default",
                worker_num=args.num_worker
            )
        worker_servicer.start_workers(restart_policy="Never")

    try:
        while True:
            if task_q.finished():
                break
            time.sleep(30)
    except KeyboardInterrupt:
        logger.warning("Server stopping")

    if args.num_worker:
        # TODO: worker_servicer.remove_workers supports synchronized call
        worker_servicer.remove_workers()
        # wait for worker pod to be deleted
        max_check_num = 10
        for _ in range(max_check_num):
            time.sleep(3)
            counters = worker_servicer.get_counters()
            if counters["pod_count"] == 0:
                break
    server.stop(0)


if __name__ == "__main__":
    logging.basicConfig()
    main()
