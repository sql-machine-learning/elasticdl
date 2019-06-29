class GRPC(object):
    # gRPC limits the size of message by default to 4MB.
    # It's too small to send model parameters.
    MAX_MESSAGE_SIZE = 256 * 1024 * 1024
