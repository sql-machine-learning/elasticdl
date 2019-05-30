# common helper methods for model manipulation.
import importlib.util


def load_user_model(model_file):
    spec = importlib.util.spec_from_file_location(model_file, model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def build_model(model, feature_columns):
    if len(feature_columns) == 1:
        # add 1 in the as the first item in input_shape tuple, as tf.keras requires this additional shape dimension.
        # https://github.com/tensorflow/tensorflow/blob/fac9d70abfb1465da53d9574173f19f235ee6d02/tensorflow/python/keras/layers/core.py#L467
        model.build((1,) + tuple(feature_columns[0].shape))
    else:
        input_shapes = []
        for f_col in feature_columns:
            input_shapes.append((1,) + tuple(f_col.shape))
        model.build(input_shapes)
