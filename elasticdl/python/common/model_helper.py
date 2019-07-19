import importlib.util
import os


def load_module(module_file):
    spec = importlib.util.spec_from_file_location(module_file, module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# "mnist_functional_api.mnist_functional_api.custom_model" -> "custom_model"
def _get_model_def_name(model_def):
    return model_def.split(".")[-1]


# "mnist_functional_api.mnist_functional_api.custom_model" -> "mnist_functional_api/mnist_functional_api.py"
def _get_model_def_file_path(model_def):
    return "/".join(model_def.split(".")[:-1]) + ".py"


def load_model_from_module(model_def, model_module, model_params):
    model_def_name = _get_model_def_name(model_def)
    if model_def_name in model_module:
        custom_model_name = model_def_name
    else:
        raise ValueError(
            "Cannot find the custom model function/class "
            "in model definition files"
        )
    if model_params:
        kvs = model_params.split(",")
        model_params_dict = {}
        for kv in kvs:
            k, v = kv.split("=")
            model_params_dict[k] = eval(v)
        return model_module[custom_model_name](**model_params_dict)
    else:
        return model_module[custom_model_name]()


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


def get_model_file(model_zoo, model_def):
    return os.path.join(model_zoo, _get_model_def_file_path(model_def))
