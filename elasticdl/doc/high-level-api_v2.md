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
class DemoModel(tf.keras.Model):

    def __init__(self, context):
        """
        Args:
            context: framework provided for model building phase, such as worker_id,
                    paths like train_data_dir, evaluation_dir
        """
        self._context = context

    def __call__(self, mode):
        """
        Args:
            inputs:
            mode: mode of current operations, e.g. Mode.TRAINING
        """
        # define outputs
        return outputs

    @property
    def loss(self):
        # defined in __call__
        return self._loss

    @property
    def optimizer(self):
        # defined in __call__
        return self._optimizer

    @property
    def metrics(self, mode="TRAIN"):
        """
        defined in __call__
        user can return different kind of metrics according to mode
        Returns:
            dict of metric
        """

        return self._metrics

    def train_op(self):
        """optimizing operations
        by default: we wil use model.optimizer and model.loss to get it.
        """
        return None

```

This interface is almost the same as model and function names but everything is in the model instance.

- default\_optimizer -> optimizer
- default\_loss -> loss

### Notes

+ `dataset_fn` is removed. User can get specific paths from context and build dataset.
+ `PredictionOutputsProcessor` will be changed to hook or callback. Currently stay unchanged.

### Changes

- Model definitions need to be changed
- Redundant command line arguments need to be removed
- Function signatures should pass model instance
