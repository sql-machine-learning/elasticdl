from collections import namedtuple

FeatureTransformInfo = namedtuple(
    "FeatureTransformInfo",
    ["name", "input_name", "output_name", "op_name", "output_dtype", "param"],
)
SchemaInfo = namedtuple("SchemaInfo", ["name", "dtype"])


class TransformOp(object):
    HASH = "HASH"
    BUCKETIZE = "BUCKETIZE"
    LOOKUP = "LOOKUP"
    EMBEDDING = "EMBEDDING"
    GROUP = "GROUP"
    ARRAY = "ARRAY"
