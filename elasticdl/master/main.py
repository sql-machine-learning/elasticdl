from concurrent import futures
import logging
import time
import argparse
import os
import importlib
from contextlib import contextmanager

import grpc
import tensorflow as tf

tf.enable_eager_execution()

from recordio import File
from proto import master_pb2_grpc
from .servicer import MasterServicer
from .task_queue import _TaskQueue


def _make_task_queue(data_dir, record_per_task, num_epoch):
    f_records = {}
    for f in os.listdir(data_dir):
        p = os.path.join(data_dir, f)
        with File(p, "r") as rio:
            f_records[p] = rio.count()
    return _TaskQueue(f_records, record_per_task, num_epoch)

def _load_user_model(model_file, model_class):
    with add_to_path(os.path.dirname(absolute_path)):
        spec = importlib.util.spec_from_file_location(absolute_path, absolute_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        user_model = getattr(module, model_class)

        model_inst = user_model()
        model_inst.build(model_inst.input_shapes())
        return model_inst

@contextmanager
def _add_to_path(p):
    import sys

    old_path = sys.path
    sys.path = sys.path[:]
    sys.path.insert(0, p)
    try:
        yield
    finally:
        sys.path = old_path

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
    return parser.parse_args()


def main():
    logger = logging.getLogger("master")
    args = _parse_args()
    task_q = _make_task_queue(
        args.train_data_dir, args.record_per_task, args.num_epoch
    )
    # TODO: use user provided optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    model_inst = _load_user_model(args.model_file, args.model_class)
    
    servicer = MasterServicer(
        logger, args.grads_to_wait, args.minibatch_size, optimizer, task_q
    )
    for variable in model_inst.trainable_variables:
        servicer.set_model_var(variable.name, variable.numpy())

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=64))
    master_pb2_grpc.add_MasterServicer_to_server(
        MasterServicer(
            logger, args.grads_to_wait, args.minibatch_size, optimizer, task_q
        ),
        server,
    )
    server.add_insecure_port("[::]:50001")
    server.start()
    logger.warning("Server started")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.warning("Server stopping")
        server.stop(0)


if __name__ == "__main__":
    logging.basicConfig()
    main()
