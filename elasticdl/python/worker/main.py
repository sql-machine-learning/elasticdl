import grpc

from elasticdl.python.common import log_util
from elasticdl.python.common.args import (
    ALL_ARGS_GROUPS,
    parse_worker_args,
    print_args,
)
from elasticdl.python.common.constants import GRPC
from elasticdl.python.worker.worker import Worker


def main():
    args = parse_worker_args()
    print_args(args, groups=ALL_ARGS_GROUPS)
    channel = grpc.insecure_channel(
        args.master_addr,
        options=[
            ("grpc.max_send_message_length", GRPC.MAX_SEND_MESSAGE_LENGTH),
            (
                "grpc.max_receive_message_length",
                GRPC.MAX_RECEIVE_MESSAGE_LENGTH,
            ),
        ],
    )

    logger = log_util.get_logger(__name__)

    logger.info("Starting worker %d", args.worker_id)
    worker = Worker(
        args.worker_id,
        args.job_type,
        args.minibatch_size,
        args.model_zoo,
        channel=channel,
        embedding_service_endpoint=eval(args.embedding_service_endpoint),
        dataset_fn=args.dataset_fn,
        loss=args.loss,
        optimizer=args.optimizer,
        eval_metrics_fn=args.eval_metrics_fn,
        model_def=args.model_def,
        model_params=args.model_params,
        get_model_steps=args.get_model_steps,
    )
    worker.run()


if __name__ == "__main__":
    main()
