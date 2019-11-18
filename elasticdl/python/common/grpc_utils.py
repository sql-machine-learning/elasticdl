import grpc

from elasticdl.python.common.constants import GRPC


def build_channel(addr):
    channel = grpc.insecure_channel(
        addr,
        options=[
            ("grpc.max_send_message_length", GRPC.MAX_SEND_MESSAGE_LENGTH),
            (
                "grpc.max_receive_message_length",
                GRPC.MAX_RECEIVE_MESSAGE_LENGTH,
            ),
        ],
    )
    return channel
