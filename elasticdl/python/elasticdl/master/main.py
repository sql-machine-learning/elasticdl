import logging
import time
import argparse
import os
import recordio

import grpc
import tensorflow as tf

tf.enable_eager_execution()

from contextlib import closing
from concurrent import futures
from elasticdl.proto import elasticdl_pb2_grpc
from elasticdl.python.elasticdl.master.servicer import MasterServicer
from elasticdl.python.elasticdl.master.task_queue import _TaskQueue
from elasticdl.python.elasticdl.master.k8s_worker_manager import WorkerManager
from elasticdl.python.elasticdl.common.model_helper import load_user_model, build_model


def _make_task_queue(data_dir, record_per_task, num_epoch):
    f_records = {}
    for f in os.listdir(data_dir):
        p = os.path.join(data_dir, f)
        with closing(recordio.Index(p)) as rio:
            f_records[p] = rio.num_records()
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
        "--checkpoint_dir",
        help="The directory to store the checkpoint files",
        default="",
    )
    parser.add_argument(
        "--checkpoint_steps",
        type=int,
        help="Save checkpoint every this many steps. If 0, no checkpoints to save.",
        default=0,
    )
    parser.add_argument(
        "--keep_checkpoint_max",
        type=int,
        help="The maximum number of recent checkpoint files to keep. If 0, keep all.",
        default=3,
    )
    parser.add_argument(
        "--worker_cpu_request",
        help="the minimal cpu required by worker in training",
        default="1000m",
    )
    parser.add_argument(
        "--worker_cpu_limit",
        help="the maximal cpu used by worker in training",
        default="1000m",
    )
    parser.add_argument(
        "--worker_memory_request",
        help="the minimal memory required by worker in training",
        default="4096Mi",
    )
    parser.add_argument(
        "--worker_memory_limit",
        help="the maximal memory used by worker in training",
        default="4096Mi",
    )
    parser.add_argument(
        "--worker_pod_priority",
        help="the requested priority of worker pod")
    parser.add_argument(
        "--worker_image", help="docker image for worker", default=None
    )
    parser.add_argument("--job_name", help="job name", required=True)
    parser.add_argument(
        "--codec_type",
        default="bytes",
        choices=["tf_example", "bytes"],
        help="Type of codec(tf_example or bytes)",
    )
    parser.add_argument("--volume_name",
        help="the volume name of network filesytem")
    parser.add_argument("--mount_path",
        help="the mount path in the docker container")
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        default="WARNING",
        help="the logging level",
    )
    parser.add_argument("--image_pull_policy",
        help="the image pull policy of master and worker")
    return parser.parse_args()


def main():
    args = _parse_args()

    # TODO: pass port via flags.
    PORT = 50001

    # Initialize logger
    logging.basicConfig(
        format='%(asctime)s %(name)s %(levelname)-8s '
        '[%(filename)s:%(lineno)d] %(message)s',
    )
    # Set level for ROOT logger.
    logging.getLogger().setLevel(args.log_level)
    logger = logging.getLogger(__name__)

    task_q = _make_task_queue(
        args.train_data_dir, args.record_per_task, args.num_epoch
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
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_steps=args.checkpoint_steps,
            keep_checkpoint_max=args.keep_checkpoint_max,
        ),
        server,
    )
    server.add_insecure_port("[::]:{}".format(PORT))
    server.start()
    logger.info("Server started at port: %d", PORT)

    if args.num_worker:
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
            args.log_level
        ]

        worker_manager = WorkerManager(
            task_q,
            job_name=args.job_name,
            worker_image=args.worker_image,
            command=worker_command,
            args=worker_args,
            namespace="default",
            num_worker=args.num_worker,
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
