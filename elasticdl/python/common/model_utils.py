import importlib.util
import os
import tensorflow as tf

from elasticdl.python.common.hash_utils import int_to_id, string_to_id
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.common.tensor import Tensor
from elasticdl.python.ps.parameters import Parameters
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


def get_dict_from_params_str(params_str):
    """Get the dictionary of kv pairs in a string separated
    by semi-colon."""
    if params_str:
        kvs = params_str.split(";")
        params_dict = {}
        for kv in kvs:
            k, v = kv.strip().split("=")
            params_dict[k] = eval(v)
        return params_dict
    else:
        return None


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


def restore_model_params_from_checkpoint(
    checkpoint_dir, shard_index, shard_num
):
    """Restore a shard parameters from the checkpoint directory.
    If shard_num=1, a entire model parameters will be restored.

    Args:
        checkpoint_dir: a directory with checkpoint files.
        shard_index: Model shard index, e.g. the PS instance index
            using ParameterServerStrategy with multiple PS instances.
        shard_num: The total number of model shards, e.g. the total PS
            instancecount using ParameterServerStrategy with multiple
            PS instances.

    Return:
        parameters: A Parameter object which contains model version,
            non-embedding parameters and embedding tables for the
            PS instance with ps_id.
    """
    from elasticdl.python.ps.embedding_table import create_embedding_table

    variable_shard_files = os.listdir(checkpoint_dir)
    non_embedding_vars = {}
    embedding_tables = {}
    version = None
    for shard_file in variable_shard_files:
        shard_file_path = os.path.join(checkpoint_dir, shard_file)
        model_pb = load_from_checkpoint_file(shard_file_path)
        if version is None:
            version = model_pb.version
        elif version != model_pb.version:
            raise ValueError(
                "The versions in model shards are not consistency"
            )

        for embedding_info_pb in model_pb.embedding_table_info:
            embedding_table = create_embedding_table(embedding_info_pb)
            embedding_tables.setdefault(embedding_table.name, embedding_table)

        (
            shard_non_embedding_vars,
            shard_embedding_table_values,
        ) = get_params_shard_from_pb(model_pb, shard_index, shard_num)
        non_embedding_vars.update(shard_non_embedding_vars)
        for name, pair in shard_embedding_table_values.items():
            embedding_tables[name].set(pair[0], pair[1])
    parameters = Parameters()
    parameters.non_embedding_params.update(non_embedding_vars)
    parameters.embedding_params.update(embedding_tables)
    parameters.version = version
    return parameters


def get_params_shard_from_pb(model_pb, shard_index, shard_num):
    """Get parameters including variables values and embedding table
    from a model protobuf.

    Args:
        model_pb: A Model protobuf instance.
        shard_index: Model shard index.
        shard_num: The total number of model shards.

    Return:
        non_embedding_vars: A Python dict in which the key is a variable
            name and the value is a `tf.Variable` object.
        embedding_table_values: A Python dict in which the key is an embedding
            table name and the value is a tuple with 2 elements. The value[0]
            is indices and value[1] is the corresponding embedding vector.
    """
    non_embedding_vars = {}
    embedding_table_values = {}

    for tensor_pb in model_pb.param:
        tensor = Tensor.from_tensor_pb(tensor_pb)
        if tensor.indices is not None:
            embedding_table_values.setdefault(tensor.name, ([], []))
            for embedding_id, vector in zip(tensor.indices, tensor.values):
                if int_to_id(embedding_id, shard_num) == shard_index:
                    embedding_table_values[tensor.name][0].append(embedding_id)
                    embedding_table_values[tensor.name][1].append(vector)
        else:
            if string_to_id(tensor.name, shard_num) == shard_index:
                non_embedding_vars[tensor.name] = tf.Variable(
                    initial_value=tensor.values,
                    trainable=True
                )
    return non_embedding_vars, embedding_table_values
