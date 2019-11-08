import grpc

from elasticdl.python.common import log_utils
from elasticdl.python.common.args import parse_worker_args
from elasticdl.python.common.constants import GRPC
from elasticdl.python.worker.worker import Worker


def main():
    args = parse_worker_args()
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

    # TODO, create PS channels here
    ps_channels = None

    logger = log_utils.get_logger(__name__)

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
        data_reader_params=args.data_reader_params,
        prediction_outputs_processor=args.prediction_outputs_processor,
        get_model_steps=args.get_model_steps,
        distribution_strategy=args.distribution_strategy,
        ps_channels=ps_channels,
    )
    worker.run()


if __name__ == "__main__":
    main()
