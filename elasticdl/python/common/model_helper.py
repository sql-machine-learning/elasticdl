import importlib.util
import os

from elasticdl.python.common.log_util import default_logger as logger
from elasticdl.python.worker.prediction_outputs_processor import (
    BasePredictionOutputsProcessor,
)


def load_module(module_file):
    spec = importlib.util.spec_from_file_location(module_file, module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model_from_module(model_def, model_module, model_params):
    model_def_name = model_def.split(".")[-1]
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


def get_module_file_path(model_zoo, spec_key):
    """Get the path to module file from model zoo and the spec string.

    For example, if `model_zoo = "model_zoo"` and
    `spec_key = "test_module.custom_model"`, the function returns
    "model_zoo/test_module.py".
    """
    return os.path.join(model_zoo, "/".join(spec_key.split(".")[:-1]) + ".py")


def _get_spec_value(spec_key, model_zoo, default_module, required=False):
    """Get the value to the given spec key.

    Notes:

    * If the dot-splitted spec key (e.g. "test_module.custom_model"
      is splitted into "test_module" and "custom_model") is of length 1
      (e.g. `spec_key` is "custom_model"), return the value in the
      specified `default_module`.
    * If the spec key does not exist in the module, return `None`.
    """
    spec_key_items = spec_key.split(".")
    spec_key_base = spec_key_items[-1]
    if len(spec_key_items) == 1:
        spec_key_module = default_module
    else:
        spec_key_module = load_module(
            get_module_file_path(model_zoo, spec_key)
        ).__dict__
    spec_value = (
        spec_key_module[spec_key_base]
        if spec_key_base in spec_key_module
        else None
    )
    if required and spec_value is None:
        raise Exception(
            "Missing required spec key %s in the module: %s"
            % (spec_key_base, spec_key)
        )
    return spec_value


def get_model_spec(
    model_zoo,
    model_def,
    model_params,
    dataset_fn,
    loss,
    optimizer,
    eval_metrics_fn,
    prediction_outputs_processor,
):
    """Get the model spec items in a tuple.

    The model spec tuple contains the following items in order:

    * The model object instantiated with parameters specified
      in `model_params`,
    * The `dataset_fn`,
    * The `loss`,
    * The `optimizer`,
    * The `eval_metrics_fn`,
    * The `prediction_outputs_processor`. Note that it will print
      warning if it's not inherited from `BasePredictionOutputsProcessor`.
    """
    model_def_module_file = get_module_file_path(model_zoo, model_def)
    default_module = load_module(model_def_module_file).__dict__
    model = load_model_from_module(model_def, default_module, model_params)
    prediction_outputs_processor = _get_spec_value(
        prediction_outputs_processor, model_zoo, default_module
    )
    if prediction_outputs_processor and not isinstance(
        prediction_outputs_processor, BasePredictionOutputsProcessor
    ):
        logger.warning(
            "prediction_outputs_processor is not "
            "inherited from BasePredictionOutputsProcessor. "
            "Prediction outputs may not be processed correctly."
        )
    return (
        model,
        _get_spec_value(dataset_fn, model_zoo, default_module, required=True),
        _get_spec_value(loss, model_zoo, default_module, required=True),
        _get_spec_value(optimizer, model_zoo, default_module, required=True),
        _get_spec_value(
            eval_metrics_fn, model_zoo, default_module, required=True
        ),
        prediction_outputs_processor,
    )


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


def find_layer(model, layer_class):
    """
    Find all layers in model that are instances of layer_class
    """
    layers = []
    for layer in model.layers:
        if isinstance(layer, layer_class):
            layers.append(layer)
        elif hasattr(layer, "layers"):
            # search in nested layers
            layers += find_layer(layer, layer_class)
    return layers
