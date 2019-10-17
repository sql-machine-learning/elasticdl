import os
import time
from concurrent import futures

import grpc
from kubernetes.client import V1EnvVar

from elasticdl.proto import elasticdl_pb2_grpc
from elasticdl.python.common.args import (
    build_arguments_from_parsed_result,
    parse_envs,
    parse_master_args,
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
from elasticdl.python.master.master import Master
from elasticdl.python.data.data_reader import create_data_reader
from elasticdl.python.elasticdl.layers.embedding import Embedding
from elasticdl.python.master.checkpoint_service import CheckpointService
from elasticdl.python.master.embedding_service import EmbeddingService
from elasticdl.python.master.evaluation_service import EvaluationService
from elasticdl.python.master.k8s_worker_manager import WorkerManager
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.master.task_dispatcher import _TaskDispatcher
from elasticdl.python.master.tensorboard_service import TensorboardService


def main():
    args = parse_master_args()
    logger = get_logger("master", level=args.log_level.upper())

    master = Master(args)
    master.run(args)


if __name__ == "__main__":
    main()
