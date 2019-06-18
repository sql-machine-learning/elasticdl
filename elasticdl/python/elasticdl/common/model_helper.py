# common helper methods for model manipulation.
import importlib.util


def load_user_model(model_file):
    spec = importlib.util.spec_from_file_location(model_file, model_file)
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
