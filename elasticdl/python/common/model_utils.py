import importlib.util
import os

from tensorflow.python.keras.callbacks import CallbackList

from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.data.odps_io import is_odps_configured
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
        model_params_dict = get_dict_from_params_str(model_params)
        return model_module[custom_model_name](**model_params_dict)
    else:
        return model_module[custom_model_name]()


def load_callbacks_from_module(callbacks_def, model_module):
    callbacks_def_name = callbacks_def.split(".")[-1]
    callbacks_fn = _get_spec_value(callbacks_def_name, None, model_module)
    callbacks = [] if callbacks_fn is None else callbacks_fn()
    return CallbackList(callbacks)


def set_callback_parameters(
    callback_list,
    batch_size=None,
    epochs=None,
    metric=None,
    saved_model_path=None,
    checkpoint_path=None,
):
    """Sets callback parameters.

    Arguments:
        callback_list: CallbackList instance.
        batch_size: Number of samples per batch
        epochs: Number of epoch to train
        metrics: Evaluation metrics
        saved_model_path: Path to export SavedModel
        checkpoint_path: Path to save checkpoint
    """
    callback_params = {
        'batch_size': batch_size,
        'epochs': epochs,
        'metric': metric,
        'saved_model_path': saved_model_path,
        'checkpoint_path': checkpoint_path
    }
    callback_list.set_params(callback_params)


def get_dict_from_params_str(params_str):
    """Get the dictionary of kv pairs in a string separated
    by semi-colon."""
    params_dict = {}
    if params_str:
        kvs = params_str.split(";")
        for kv in kvs:
            splitted = kv.strip().split("=")
            k = splitted[0]
            # if there is '=' in value, need to restore it.
            v = "=".join(splitted[1:])
            try:
                params_dict[k] = eval(v)
            except Exception:
                params_dict[k] = v
    return params_dict


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
    custom_data_reader,
    callbacks,
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
    * The `custom_data_reader`
    * The `callbacks`
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

    # If ODPS data source is used, dataset_fn is optional
    dataset_fn_required = not is_odps_configured()
    callbacks_list = load_callbacks_from_module(callbacks, default_module)

    return (
        model,
        _get_spec_value(
            dataset_fn, model_zoo, default_module, required=dataset_fn_required
        ),
        _get_spec_value(loss, model_zoo, default_module, required=True),
        _get_spec_value(optimizer, model_zoo, default_module, required=True),
        _get_spec_value(
            eval_metrics_fn, model_zoo, default_module, required=True
        ),
        prediction_outputs_processor,
        _get_spec_value(custom_data_reader, model_zoo, default_module),
        callbacks_list,
    )


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


def get_non_embedding_trainable_vars(model, embedding_layers):
    """
    Get trainable variables which are not from ElasticDL embedding layers.
    """
    embedding_items = []
    for layer in embedding_layers:
        embedding_items.extend(layer.trainable_variables)
    non_embedding_trainable_vars = []
    for var in model.trainable_variables:
        is_embedding_item = False
        for embedding_item in embedding_items:
            if var is embedding_item:
                is_embedding_item = True
                break
        if not is_embedding_item:
            non_embedding_trainable_vars.append(var)
    return non_embedding_trainable_vars
