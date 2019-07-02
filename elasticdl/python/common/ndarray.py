import numpy as np

from elasticdl.proto import elasticdl_pb2


def tensor_to_ndarray(tensor_pb):
    """
    Create an ndarray from Tensor proto message. Note: upon return, the input
    tensor message is reset and underlying buffer passed to the returned
    ndarray.
    """

    if not tensor_pb.dim:
        raise ValueError("Tensor PB has no dim defined")

    # Check that the buffer size agrees with dimensions.
    size = 4  # A float32 item occupies 4 bytes
    for d in tensor_pb.dim:
        size *= d
    if size != len(tensor_pb.content):
        raise ValueError(
            "Tensor PB size mismatch, dim: %s, len(content): %d",
            tensor_pb.dim,
            len(tensor_pb.content),
        )
    arr = np.ndarray(
        shape=tensor_pb.dim, dtype=np.float32, buffer=tensor_pb.content
    )
    tensor_pb.Clear()

    return arr


def ndarray_to_tensor(arr):
    """Convert ndarray to Tensor PB"""

    if arr.dtype != np.float32:
        raise ValueError(
            "expected ndarray to be of float32 type, got %s type", arr.dtype
        )
    tensor = elasticdl_pb2.Tensor()
    tensor.dim.extend(arr.shape)
    tensor.content = arr.tobytes()

    return tensor
