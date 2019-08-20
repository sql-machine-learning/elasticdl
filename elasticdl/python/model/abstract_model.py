from abc import ABC, abstractmethod

import tensorflow as tf

from elasticdl.python.common.constants import Mode


class ElasticDLKerasBaseModel(tf.keras.Model, ABC):
    """Base class for Keras Model used in elasticdl
    User should inherit from this class in order to work with elasticdl

    ```python
    class AwesomeModel(ElasticDLKerasBaseModel):
        def __init__(self, context=None):
            super(AwesomeModel, self).__init__(context)
    ```

    Constructor should have a keyword argument context. It can have other
    keyword arguments, those will be passed to tf.keras.Model's constructor

    ```python
    class AwesomeModel(ElasticDLKerasBaseModel):
        def __init__(self, context=None, name="X", params="xx"):
            super(AwesomeModel, self).__init__(context=context,
                                               name=name,
                                               params=params)
    ```
    """

    def __init__(self, *args, context=None, **kwargs):
        """
        Args:
            context: dict, from args and model_params
        """
        super(ElasticDLKerasBaseModel, self).__init__(*args, **kwargs)
        self._context = context or {}

    def get_model(self):
        """
        Used to unify model description of functional API and subclass
        For subclass, just return self, default implementation
        For functional API, YOU MUST return model instance created
        """
        return self

    @abstractmethod
    def optimizer(self, lr=0.1):
        """
        Return optimizer instance
        """
        pass

    @abstractmethod
    def loss(self, outputs=None, labels=None):
        """
        Return loss tensor
        """
        pass

    @abstractmethod
    def metrics(
        self, mode=Mode.TRAINING, outputs=None, predictions=None, labels=None
    ):
        """
        Return dict of metrics tensor according to mode
        """
        pass

    @abstractmethod
    def call(self, inputs, training=False):
        """
        Args:
            inputs:
            training:
        """
        pass
