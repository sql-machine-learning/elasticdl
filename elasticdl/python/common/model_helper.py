# common helper methods for model manipulation.
import importlib.util
import os


# TODO: This is not only used for model. Maybe we should put it into another
# file
def load_module(module_file):
    spec = importlib.util.spec_from_file_location(module_file, module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_checkpoint_to_file(pb_model, file_name):
    encoded_model = pb_model.SerializeToString()
    with open(file_name, "wb") as f:
        f.write(encoded_model)


def load_from_checkpoint_file(file_name):
    from elasticdl.proto import elasticdl_pb2

    pb_model = elasticdl_pb2.Model()
    with open(file_name, "rb") as f:
        pb_model.ParseFromString(f.read())
    return pb_model


def get_model_file(model_def):
    return os.path.join(model_def, os.path.basename(model_def) + ".py")
