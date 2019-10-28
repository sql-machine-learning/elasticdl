import abc

import tensorflow as tf

from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.elasticdl.layers.embedding import Embedding


class ModelHander(metaclass=abc.ABCMeta):
    """
    Generate the model to train in ElasticDL for different distributed
    strategies and export trained model in ElasticDL to SavedModel.
    """

    @abc.abstractmethod
    def generate_train_model_for_elasticdl(self, model):
        """
        Generate a model to train in ElasticDL.
        """
        pass

    @abc.abstractmethod
    def get_saved_model_from_trained_model(self, model, dataset):
        """
        Get the model which can be exported a SavedModel
        by tf.saved_model.save.
        """
        pass

    @classmethod
    def get_model_handler(cls, distribution_strategy=None):
        """
        Create a model handler to process the model for the
        distributed strategy.
        """
        if distribution_strategy == "ParameterServerStrategy":
            return ParameterServerModelHandler()
        else:
            return DefaultModelHandler()


class DefaultModelHandler(ModelHander):
    """
    Return the origin model to train and export.
    """

    def generate_train_model_for_elasticdl(self, model):
        return model

    def get_saved_model_from_trained_model(self, model, dataset):
        if not model.inputs:
            model._build_model_with_inputs(inputs=dataset, targets=None)
        return model


class ParameterServerModelHandler(ModelHander):
    def generate_train_model_for_elasticdl(self, model):
        """
        Replace the tf.keras.layers.Embedding layer in the model with
        an elasticdl.layers.Embedding layer in ParameterServerStrategy.
        """
        if type(model) == tf.keras.Sequential or model._is_graph_network:
            model = self._clone_model_for_sequential_and_functional(model)
        else:
            model = self._replace_embedding_attribute_for_subclass(model)
        return model

    def get_saved_model_from_trained_model(self, model, dataset):
        """
        To export model for tf-serving by tf.saved_model.save:
        1. Add inputs and outputs to the model.
        2. Restore Keras model and replace embedding parameters with trained
        model.
        """
        # TODO
        pass

    def _clone_model_for_sequential_and_functional(self, model):
        """
        Clone a new model with elasticdl.layers.Embedding for
        Sequential and functional API model.
        """

        def _clone_function(layer):
            if type(layer) == tf.keras.layers.Embedding:
                logger.info(
                    "Replace Keras Embedding with \
                            ElasticDL Embedding"
                )
                edl_embedding_layer = Embedding(
                    output_dim=layer.output_dim,
                    embedding_initializer=layer.embeddings_initializer,
                    mask_zero=layer.mask_zero,
                    input_length=layer.input_length,
                )
                return edl_embedding_layer
            return layer

        return tf.keras.models.clone_model(
            model, clone_function=_clone_function
        )

    def _replace_embedding_attribute_for_subclass(self, model):
        """
        Replace the keras embedding attribute with
        elasticdl.layers.Embedding layer.
        """
        for attr_name, attr_value in model.__dict__.items():
            if type(attr_value) == tf.keras.layers.Embedding:
                setattr(model, attr_name, Embedding(attr_value.output_dim))
        return model
