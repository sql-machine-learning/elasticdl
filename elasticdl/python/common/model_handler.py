import abc

import numpy as np
import tensorflow as tf

from elasticdl.python.common.constants import DistributionStrategy
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.common.save_utils import CheckpointSaver
from elasticdl.python.elasticdl.layers.embedding import Embedding
from elasticdl.python.keras.layers import SparseEmbedding
from elasticdl.python.ps.embedding_table import EmbeddingTable


def _get_trained_params_from_checkpoint(checkpoint_dir):
    """Get parameters from a checkpoint directory saved by ElasticDL"""
    parameters = CheckpointSaver.restore_params_from_checkpoint(
        checkpoint_dir, 0, 1
    )

    trained_params = parameters.non_embedding_params
    for name, table in parameters.embedding_params.items():
        # The name of variable in a tf.keras.layers.Embedding layer is
        # "{layer_name}/embeddings:0"
        var_name = name + "/embeddings:0"
        trained_params[var_name] = table
    return trained_params


def _convert_embedding_table_to_array(embedding_table, embedding_shape):
    """Convert an embedding table to a np.ndarray which can be assigned
    to trainable weights in keras embedding layers.

    Args:
        embedding_table: A `EmbeddingTable` instance.
        embedding_shape: a tuple with two elements

    Returns:
        A np.ndarray
    """
    embedding_ids = list(embedding_table.embedding_vectors.keys())
    embedding_values = list(embedding_table.embedding_vectors.values())
    embedding_weights = np.zeros(embedding_shape)
    embedding_weights[embedding_ids] = embedding_values
    return embedding_weights


class ModelHandler(metaclass=abc.ABCMeta):
    """Generate the model to train in ElasticDL for different distributed
    strategies and export trained model in ElasticDL to SavedModel.
    """

    @abc.abstractmethod
    def get_model_to_train(self, model):
        """Generate a model to train in ElasticDL.

        Args:
            model: A native keras model instance.

        Returns:
            A keras model instance for ElasticDL training.
        """

    @abc.abstractmethod
    def get_model_to_export(self, model, dataset):
        """Get the model which can be exported a SavedModel
        by tf.saved_model.save.

        Args:
            model: A keras model instance trained by ElasticDL and
            it may contains `elasticdl.layers.Embedding` layers.
            dataset: A `tf.data.Dataset` instance which has the same outputs as
                the training dataset.

        Returns:
            A keras model instance trained by ElasticDL.
        """

    @classmethod
    def get_model_handler(
        cls, distribution_strategy=None, checkpoint_dir=None
    ):
        """Create a model handler to process the model for the
        distributed strategy.

        Args:
            distribution_strategy (string): distribution strategy name
            checkpoint_dir: Checkpoint directory to save model parametes
                during training.

        Return:
            ModelHandler subclass instance.
        """
        if distribution_strategy == DistributionStrategy.PARAMETER_SERVER:
            return ParameterServerModelHandler(checkpoint_dir=checkpoint_dir)
        elif distribution_strategy == DistributionStrategy.ALLREDUCE:
            logger.warning(
                "Allreduce distribution strategy is not supported yet. "
                "Switching to use the default distribution strategy."
            )
        return DefaultModelHandler()


class DefaultModelHandler(ModelHandler):
    """Return the origin model to train and export."""

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
    """Model handler for parameter server strategy.
    For training, The handler will replace `tf.keras.layers.Embedding`
    layers with`elasticdl.layers.Embedding` for training.
    For saving model, the handler will restore Keras model definition and
    pull trained parameters from parameter server(s) for the model.
    """

    def __init__(self, checkpoint_dir=None):
        """
        Arguments:
            checkpoint_dir: A checkpoint directory to save all model
                parameters during training.
        """
        self._checkpoint_dir = checkpoint_dir

    def get_model_to_train(self, model):
        """Replace the tf.keras.layers.Embedding layer in the model with
        an elasticdl.layers.Embedding layer in ParameterServerStrategy.
        """
        if type(model) == tf.keras.Sequential or model._is_graph_network:
            model = self._clone_model_with_edl_embedding(model)
        else:
            model = self._replace_attr_with_edl_embedding(model)
        return model

    def get_model_to_export(self, model, dataset):
        """Get the model which can be exported to a SavedModel by
        `tf.saved_model.save`.
        """
        model = self._restore_keras_model_def(model)
        if not model.inputs:
            # build model to add inputs and outputs that
            # can be consumed by tf-serving
            model._build_model_with_inputs(inputs=dataset, targets=None)

        checkpoint_dir = CheckpointSaver.get_valid_lastest_version_dir(
            self._checkpoint_dir
        )
        if checkpoint_dir is None:
            logger.warning("No available checkpoint to export model")
            return model

        trained_params = _get_trained_params_from_checkpoint(checkpoint_dir)
        for var in model.trainable_variables:
            if isinstance(trained_params[var.name], EmbeddingTable):
                embedding_params = _convert_embedding_table_to_array(
                    trained_params[var.name], var.shape
                )
                var.assign(embedding_params)
            else:
                var.assign(trained_params[var.name].numpy())
        return model

    def _restore_keras_model_def(self, model):
        """Restore Keras model definition by replacing
        `elasticdl.layers.Embedding` layers with
        `tf.keras.layers.Embedding` layers.
        """
        # clear keras model session to avoid clutter from old models/layers.
        tf.keras.backend.clear_session()
        if (
            isinstance(model, tf.keras.models.Model)
            and not model._is_graph_network
        ):
            model = self._replace_attr_with_keras_embedding(model)
        else:
            model = self._clone_model_with_keras_embedding(model)
        return model

    @staticmethod
    def _clone_model_with_edl_embedding(model):
        """Clone a new model and replace keras embedding layers including
        `tf.keras.layers.Embedding` and `SparseEmbedding` with
        `elasticdl.layers.Embedding`
        """

        def _clone_function(layer):
            if type(layer) in [tf.keras.layers.Embedding, SparseEmbedding]:
                logger.debug(
                    "Replace {} with {}".format(layer.name, Embedding)
                )
                # ElasticDL embedding only accept a string type initializer
                init = tf.keras.initializers.serialize(
                    layer.embeddings_initializer
                )["class_name"]

                embedding_layer = Embedding(
                    output_dim=layer.output_dim,
                    input_dim=layer.input_dim,
                    embeddings_initializer=init,
                    mask_zero=layer.mask_zero,
                    input_length=layer.input_length,
                    name=layer.name,
                )
                if type(layer) == SparseEmbedding:
                    embedding_layer.combiner = layer.combiner
                return embedding_layer
            return layer

        return tf.keras.models.clone_model(
            model, clone_function=_clone_function
        )

    @staticmethod
    def _clone_model_with_keras_embedding(model):
        """Clone a new model and replace the `elasticdl.layers.Embedding`
        layers with `tf.keras.layers.Embedding` or `SparseEmbedding` layers
        """

        def _clone_function(layer):
            if type(layer) == Embedding:
                # ElasticDL embedding only accept a string type initializer
                if layer.combiner is not None:
                    embedding_layer = SparseEmbedding(
                        output_dim=layer.output_dim,
                        input_dim=layer.input_dim,
                        embeddings_initializer=layer.embeddings_initializer,
                        mask_zero=layer.mask_zero,
                        input_length=layer.input_length,
                        name=layer.name,
                        combiner=layer.combiner,
                    )
                else:
                    embedding_layer = tf.keras.layers.Embedding(
                        output_dim=layer.output_dim,
                        input_dim=layer.input_dim,
                        embeddings_initializer=layer.embeddings_initializer,
                        mask_zero=layer.mask_zero,
                        input_length=layer.input_length,
                        name=layer.name,
                    )
                return embedding_layer
            return layer

        return tf.keras.models.clone_model(
            model, clone_function=_clone_function
        )

    @staticmethod
    def _replace_attr_with_edl_embedding(model):
        """Replace the keras embedding attributes in the model with
        `elasticdl.layers.Embedding` layers.
        """
        for name, value in model.__dict__.items():
            if type(value) in [tf.keras.layers.Embedding, SparseEmbedding]:
                init = tf.keras.initializers.serialize(
                    value.embeddings_initializer
                )["class_name"]
                embedding_layer = Embedding(
                    output_dim=value.output_dim,
                    input_dim=value.input_dim,
                    embeddings_initializer=init,
                    mask_zero=value.mask_zero,
                    input_length=value.input_length,
                )
                if type(value) == SparseEmbedding:
                    embedding_layer.combiner = value.combiner
                setattr(model, name, embedding_layer)
        return model

    @staticmethod
    def _replace_attr_with_keras_embedding(model):
        """Replace the elasticdl.layers.Embedding attributes in the model
        with `tf.keras.layers.Embedding` or `SparseEmbedding` layers.
        """
        for name, value in model.__dict__.items():
            if type(value) == Embedding:
                if value.combiner is not None:
                    embedding_layer = SparseEmbedding(
                        output_dim=value.output_dim,
                        input_dim=value.input_dim,
                        embeddings_initializer=value.embeddings_initializer,
                        mask_zero=value.mask_zero,
                        input_length=value.input_length,
                        combiner=value.combiner,
                    )
                else:
                    embedding_layer = tf.keras.layers.Embedding(
                        output_dim=value.output_dim,
                        input_dim=value.input_dim,
                        embeddings_initializer=value.embeddings_initializer,
                        mask_zero=value.mask_zero,
                        input_length=value.input_length,
                    )
                setattr(model, name, embedding_layer)
        return model
