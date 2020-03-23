from enum import Enum


class TransformOpType(Enum):
    HASH = (1,)
    BUCKETIZE = (2,)
    LOOKUP = (3,)
    EMBEDDING = (4,)
    CONCAT = (5,)
    ARRAY = 6


class TransformOp(object):
    def __init__(self, name, input, output):
        self.name = name
        # The input name or a list of input names
        self.input = input
        # The output name
        self.output = output
        # The type of the TransformOp.
        # The value is one of `TransformOpType`.
        self.op_type = None


class Vocabularize(TransformOp):
    def __init__(
        self, name, input, output, vocabulary_list=None, vocabulary_file=None
    ):
        super(Vocabularize, self).__init__(name, input, output)
        self.op_type = TransformOpType.LOOKUP
        self.vocabulary_list = vocabulary_list
        self.vocabulary_file = vocabulary_file


class Bucketize(TransformOp):
    def __init__(self, name, input, output, num_buckets=None, boundaries=None):
        super(Bucketize, self).__init__(name, input, output)
        self.op_type = TransformOpType.BUCKETIZE
        self.num_buckets = num_buckets
        self.boundaries = boundaries


class Embedding(TransformOp):
    def __init__(self, name, input, output, input_dim, output_dim):
        super(Embedding, self).__init__(name, input, output)
        self.op_type = TransformOpType.EMBEDDING
        self.input_dim = input_dim
        self.output_dim = output_dim


class Hash(TransformOp):
    def __init__(self, name, input, output, hash_bucket_size):
        super().__init__(name, input, output)
        self.op_type = TransformOpType.HASH
        self.hash_bucket_size = hash_bucket_size


class Concat(TransformOp):
    def __init__(self, name, input, output, id_offsets):
        super().__init__(name, input, output)
        self.op_type = TransformOpType.CONCAT
        self.id_offsets = id_offsets


class Array(TransformOp):
    def __init__(self, name, input, output):
        super().__init__(name, input, output)
        self.op_type = TransformOpType.ARRAY


class SchemaInfo(object):
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype
