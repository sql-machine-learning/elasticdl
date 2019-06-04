# common helper methods for model manipulation.
import importlib.util
from elasticdl.proto import elasticdl_pb2


def load_user_model(model_file):
    spec = importlib.util.spec_from_file_location(model_file, model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def build_model(model, feature_columns):
    if len(feature_columns) == 1:
        # add 1 as the first item in input_shape tuple, as tf.keras requires this additional shape dimension.
        # https://github.com/tensorflow/tensorflow/blob/fac9d70abfb1465da53d9574173f19f235ee6d02/tensorflow/python/keras/layers/core.py#L467
        model.build((1,) + tuple(feature_columns[0].shape))
    else:
        input_shapes = [(1,) + tuple(f_col.shape) for f_col in feature_columns]
        model.build(input_shapes)

def save_checkpoint_to_file(pb_model, file_name):
    encoded_model = pb_model.SerializeToString()
    with open(file_name, "wb") as f:
        f.write(encoded_model)

def load_from_checkpoint_file(file_name):
    pb_model = elasticdl_pb2.Model()
    with open(file_name, "rb") as f:
        pb_model.ParseFromString(f.read())
    return pb_model
    
