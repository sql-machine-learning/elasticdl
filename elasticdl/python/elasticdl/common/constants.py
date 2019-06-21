class GRPCOptions(object):
    # gRPC limits the size of message by default to 4MB. 
    # It's too small to send model parameters.
    GRPC_MAX_SEND_MESSAGE_LENGTH = 256 * 1024 * 1024
    GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 256 * 1024 * 1024
