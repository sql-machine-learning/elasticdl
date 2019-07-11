import importlib.util
import os

# The default custom model name in model definition file
DEFAULT_FUNCTIONAL_CUSTOM_MODEL_NAME = "custom_model"
DEFAULT_SUBCLASS_CUSTOM_MODEL_NAME = "CustomModel"


def load_module(module_file):
    spec = importlib.util.spec_from_file_location(module_file, module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model_from_module(model_class, model_module):
    if model_class in model_module:
        custom_model_name = model_class
    elif DEFAULT_FUNCTIONAL_CUSTOM_MODEL_NAME in model_module:
        custom_model_name = DEFAULT_FUNCTIONAL_CUSTOM_MODEL_NAME
    elif DEFAULT_SUBCLASS_CUSTOM_MODEL_NAME in model_module:
        custom_model_name = DEFAULT_SUBCLASS_CUSTOM_MODEL_NAME
    else:
        raise ValueError(
            "Cannot find the custom model function/class "
            "in model definition files"
        )
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


def get_model_file(model_def):
    return os.path.join(model_def, os.path.basename(model_def) + ".py")
