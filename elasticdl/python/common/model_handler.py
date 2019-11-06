import abc

import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common import model_utils
from elasticdl.python.common.constants import DistributionStrategy
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.common.tensor import Tensor
from elasticdl.python.elasticdl.layers.embedding import Embedding


class ModelHandler(metaclass=abc.ABCMeta):
    """
    Generate the model to train in ElasticDL for different distributed
    strategies and export trained model in ElasticDL to SavedModel.
    """

    @abc.abstractmethod
    def get_model_to_train(self, model):
        """
        Generate a model to train in ElasticDL.
        """
        pass

    @abc.abstractmethod
    def get_model_to_export(self, model, dataset):
        """
        Get the model which can be exported a SavedModel
        by tf.saved_model.save.
        """
        pass

    @classmethod
    def get_model_handler(cls, distribution_strategy=None, stub=None):
        """
        Create a model handler to process the model for the
        distributed strategy.
        """
        if distribution_strategy == DistributionStrategy.PARAMETER_SERVER:
            return ParameterServerModelHandler(stub=stub)
        else:
            return DefaultModelHandler()


class DefaultModelHandler(ModelHandler):
    """
    Return the origin model to train and export.
    """

    def get_model_to_train(self, model):
        return model

    def get_model_to_export(self, model, dataset):
        """
        Get model with inputs and trained parameters to export.
        """
        if not model.inputs:
            model._build_model_with_inputs(inputs=dataset, targets=None)
        return model


class ParameterServerModelHandler(ModelHandler):
    def __init__(self, stub=None):
        self._stub = stub

    def get_model_to_train(self, model):
        """
        Replace the tf.keras.layers.Embedding layer in the model with
        an elasticdl.layers.Embedding layer in ParameterServerStrategy.
        """
        if type(model) == tf.keras.Sequential or model._is_graph_network:
            model = self._replace_embedding_layer_to_clone_model(
                model, tf.keras.layers.Embedding, Embedding
            )
        else:
            model = self._replace_embedding_attribute_for_subclass(
                model, tf.keras.layers.Embedding, Embedding
            )
        return model

    def get_model_to_export(self, model, dataset):
        """
        Get the model which can be exported to a SavedModel by
        `tf.saved_model.save`.
        """
        model = self._restore_keras_model_def(model)
        if not model.inputs:
            # build model to add inputs and outputs that
            # can be consumed by tf-serving
            model._build_model_with_inputs(inputs=dataset, targets=None)

        trained_params = self._get_trained_params(model)
        for var in model.trainable_variables:
            var.assign(trained_params[var.name])
        return model

    def _restore_keras_model_def(self, model):
        """
        Restore Keras model definition by replacing
        `elasticdl.layers.Embedding` layers with
        `tf.keras.layers.Embedding` layers.
        """
        # clear keras model session to avoid clutter from old models/layers.
        tf.keras.backend.clear_session()
        if (
            type(model) == tf.keras.models.Model
            and not model._is_graph_network
        ):
            model = self._replace_embedding_attributes_for_subclass(
                model, Embedding, tf.keras.layers.Embedding
            )
        else:
            model = self._replace_embedding_layer_to_clone_model(
                model, Embedding, tf.keras.layers.Embedding
            )
        return model

    def _replace_embedding_layer_to_clone_model(
        self, model, src_embedding_class, dst_embedding_class
    ):
        """
        Clone a new model by cloning model and replace the
        src_embedding_class layer with a dst_embedding_class.
        """

        def _clone_function(layer):
            if type(layer) == src_embedding_class:
                logger.debug(
                    "Replace {} with {}".format(
                        src_embedding_class, dst_embedding_class
                    )
                )
                embedding_layer = dst_embedding_class(
                    output_dim=layer.output_dim,
                    input_dim=layer.input_dim,
                    embeddings_initializer=layer.embeddings_initializer,
                    mask_zero=layer.mask_zero,
                    input_length=layer.input_length,
                )
                return embedding_layer
            return layer

        return tf.keras.models.clone_model(
            model, clone_function=_clone_function
        )

    def _replace_embedding_attributes_for_subclass(
        self, model, src_embedding_class, dst_embedding_class
    ):
        """
        Replace the keras embedding attribute with
        elasticdl.layers.Embedding layer.
        """
        for name, value in model.__dict__.items():
            if type(value) == src_embedding_class:
                embedding_layer = dst_embedding_class(
                    output_dim=value.output_dim,
                    input_dim=value.input_dim,
                    embeddings_initializer=value.embeddings_initializer,
                    mask_zero=value.mask_zero,
                    input_length=value.input_length,
                )
                setattr(model, name, embedding_layer)
        return model

    def _get_trained_params(self, model):
        """
        get all trained variable values of the model
        """
        trained_params = self._get_non_embedding_variables(
            -1, elasticdl_pb2.MINIMUM
        )
        trained_embedding_params = self._get_trained_embedding_params(model)
        trained_params.update(trained_embedding_params)
        return trained_params

    def _get_trained_embedding_params(self, model):
        """
        Get trained embedding table from PS
        """
        embedding_params = {}
        embedding_layers = model_utils.find_layer(model, Embedding)
        for embedding_layer in embedding_layers:
            # TODO get all embedding vectors of the embedding layer from PS
            pass
        return embedding_params

    def _get_non_embedding_variables(self, version, method):
        """
        get model from master, and update model_version
        """
        req = elasticdl_pb2.GetModelRequest()
        req.version = version
        req.method = method
        model = self._stub.GetModel(req, None)
        variables = {}
        for tensor_pb in model.param:
            tensor = Tensor.from_tensor_pb(tensor_pb)
            variables[tensor.name] = tensor.to_ndarray()
        return variables
