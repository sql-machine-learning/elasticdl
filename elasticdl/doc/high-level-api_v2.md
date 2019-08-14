### Introduction

High-level API is the bridge between framework and user defined code. User implements the interfaces required by framework and framework does its job accordingly.

### Current Design

[High Level API Design](https://github.com/wangkuiyi/elasticdl/blob/develop/elasticdl/doc/high-level-api.md)

Currently we use a model module and functions in module to represent a model. For example:

```python
class DemoModel(tf.keras.Model):
    def call(inputs, training=False):
        # define outputs
        return outputs

def default_loss(outputs, labels):
    # define loss
    return loss

def default_optimizer(lr=0.1):
    # define optimizer
    return optimizer

## other function definitions
```

function names can be overridden by arguments passed to worker. For example:

```bash
--loss awesome_loss
```

It has following cons:

- Whenever we need other functions:
  -  we need add corresponding default names and command line arguments.
    - You may argue that we do not have to add a command line argument if user reuses the default name.
  - certain function signatures need to be changed accordingly.
- Functions are module functions and their signatures are predefined. For example if the loss function needs additional information besides outputs and labels, we have to create global variables then.
- We cannot enforce certain functions to be defined by a clear interface. Instead we have to provide documentations for it.
- It is easy to pass a model instance with clear properties and member functions in mind  instead of model and a lot of names which user can use arbitrary strings.

### New design

Encapsulate into a single model class.


```python
from abc import ABC
from abc import abstractmethod

import tensorflow as tf

class ElasticDLKerasModelBase(tf.keras.Model, ABC):

    def __init__(self, context, *args, **kwargs):
        """
        Args:
            context: framework provided for model building phase, such as worker_id,
                    paths like train_data_dir, evaluation_dir
        """
        self._context = context
        super(ElasticDLKerasModelBase, self).__init__(*args, **kwargs)

    @abstractmethod
    def call(self, inputs, training=False):
        """
        Args:
            inputs:
            training: training phase indication
        """
        # define outputs

    @abstractmethod
    def loss(self, outputs=None, labels=None):
        """
        Return loss tensor
        """

    @abstractmethod
    def metrics(self,
                mode=Mode.TRAINING,
                outputs=None,
                predictions=None,
                labels=None,):
        """
        Return dict of metrics tensor according to mode
        """

    @abstractmethod
    def optimizer(self, lr=0.1):
        """Define optimizer"""

    def train_op(self):
        """optimizing operations
        by default: we wil use model.optimizer and model.loss to get it.
        """
        return None

```

This interface is almost the same as model and function names but everything is in the model instance.

- default\_optimizer -> optimizer
- default\_loss -> loss
- eval\_metrics\_fn -> metrics

User's model should inherit from `ElasticDLKerasModelBase`.

For example:

```python
import tensorflow as tf

from elasticdl.python.common.constants import Mode

from elasticdl.python.model import ElasticDLKerasModelBase


class CustomModel(ElasticDLKerasModelBase):
    def __init__(self, channel_last=True, context=None):
        super(CustomModel, self).__init__(context, name="cifar10_model")

        self._context = context
        use_bias = True
        self._conv_1 = tf.keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            padding="same",
            use_bias=use_bias,
            activation=None,
        )
        self._bn_1 = tf.keras.layers.BatchNormalization(
            epsilon=1e-06, axis=-1, momentum=0.9
        )
        self._relu_1 = tf.keras.layers.Activation(tf.nn.relu)

        self._conv_2 = tf.keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            padding="same",
            use_bias=use_bias,
            activation=None,
        )
        self._bn_2 = tf.keras.layers.BatchNormalization(
            epsilon=1e-06, axis=-1, momentum=0.9
        )
        self._relu_2 = tf.keras.layers.Activation(tf.nn.relu)

        self._max_pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self._dropout_1 = tf.keras.layers.Dropout(0.2)

        self._conv_3 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            use_bias=use_bias,
            activation=None,
        )
        self._bn_3 = tf.keras.layers.BatchNormalization(
            epsilon=1e-06, axis=-1, momentum=0.9
        )
        self._relu_3 = tf.keras.layers.Activation(tf.nn.relu)

        self._conv_4 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            use_bias=use_bias,
            activation=None,
        )
        self._bn_4 = tf.keras.layers.BatchNormalization(
            epsilon=1e-06, axis=-1, momentum=0.9
        )
        self._relu_4 = tf.keras.layers.Activation(tf.nn.relu)

        self._max_pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self._dropout_2 = tf.keras.layers.Dropout(0.3)

        self._conv_5 = tf.keras.layers.Conv2D(
            128,
            kernel_size=(3, 3),
            padding="same",
            use_bias=use_bias,
            activation=None,
        )
        self._bn_5 = tf.keras.layers.BatchNormalization(
            epsilon=1e-06, axis=-1, momentum=0.9
        )
        self._relu_5 = tf.keras.layers.Activation(tf.nn.relu)

        self._conv_6 = tf.keras.layers.Conv2D(
            128,
            kernel_size=(3, 3),
            padding="same",
            use_bias=use_bias,
            activation=None,
        )
        self._bn_6 = tf.keras.layers.BatchNormalization(
            epsilon=1e-06, axis=-1, momentum=0.9
        )
        self._relu_6 = tf.keras.layers.Activation(tf.nn.relu)

        self._max_pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self._dropout_3 = tf.keras.layers.Dropout(0.4)

        self._flatten_1 = tf.keras.layers.Flatten()
        self._dense_1 = tf.keras.layers.Dense(10, name="output")

    def call(self, inputs, training=False):
        x = self._conv_1(inputs["image"])
        x = self._bn_1(x)
        x = self._relu_1(x)
        x = self._conv_2(x)
        x = self._bn_2(x)
        x = self._relu_2(x)
        x = self._max_pool_1(x)
        x = self._dropout_1(x)
        x = self._conv_3(x)
        x = self._bn_3(x)
        x = self._relu_3(x)
        x = self._conv_4(x)
        x = self._bn_4(x)
        x = self._relu_4(x)
        x = self._max_pool_2(x)
        x = self._dropout_2(x)
        x = self._conv_5(x)
        x = self._bn_5(x)
        x = self._relu_5(x)
        x = self._conv_6(x)
        x = self._bn_6(x)
        x = self._relu_6(x)
        x = self._max_pool_3(x)
        x = self._dropout_3(x)
        x = self._flatten_1(x)
        return self._dense_1(x)

    def loss(self, output, labels):
        labels = tf.reshape(labels, [-1])
        return tf.reduce_mean(
            input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=output, labels=labels
            )
        )

    def optimizer(self, lr=0.1):
        return tf.optimizers.SGD(lr)

    def metrics(self,
                mode=Mode.TRAINING,
                outputs=None,
                predictions=None,
                labels=None,):

        if mode == Mode.EVALUATION:
            labels = tf.reshape(labels, [-1])
            return {
                "accuracy": tf.reduce_mean(
                        input_tensor=tf.cast(
                        tf.equal(
                            tf.argmax(predictions, 1, output_type=tf.dtypes.int32),
                            labels,
                        ),
                        tf.float32,
                    )
                )
            }
        else:
            return {}


def dataset_fn(dataset, mode):
    if mode == Mode.PREDICTION:
        raise Exception(
            "dataset_fn in prediction mode is not "
            "implemented for this model yet."
        )

    def _parse_data(record):
        feature_description = {
            "image": tf.io.FixedLenFeature([32, 32, 3], tf.float32),
            "label": tf.io.FixedLenFeature([1], tf.int64),
        }
        r = tf.io.parse_single_example(record, feature_description)
        label = r["label"]
        image = r["image"]
        return image, label

    dataset = dataset.map(_parse_data)
    dataset = dataset.map(
        lambda x, y: (
            {"image": tf.math.divide(tf.cast(x, tf.float32), 255.0)},
            tf.cast(y, tf.int32),
        )
    )
    if mode != Mode.PREDICTION:
        dataset = dataset.shuffle(buffer_size=1024)
    return dataset


```

### Notes

+ `dataset_fn` is removed. User can get specific paths from context and build dataset.
+ `PredictionOutputsProcessor` will be changed to hook or callback. Currently stay unchanged.

### Changes

- Model definitions need to be changed
- Redundant command line arguments need to be removed
- Function signatures should pass model instance
