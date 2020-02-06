# Design to Support Keras Callback

This document describes the design for supporting callback to customize the behavior of model during training, evaluation and inference in ElasticDL.

## Motivation

In deep learning, we generally need to customize the behavior of model during training, evaluation or
inference, including reading/changing the model. According to the call frequency, we may perform the behavior per batch, per epoch or at the start and end of job. `tf.keras.callbacks.Callback` is an abstract base class and has methods to perform the behavior at different call frequency, such as `on_bath_end`, `on_epoch_end` and so on. So, we adopt the interfaces of `tf.keras.callbacks.Callback` for users to customize the behavior of model in ElasticDL. 

Now we have implemented some modules which are similar to callback in ElasticDL such as `LearingRateScheduler` and `PredictionOutputsProcessor`. And user should write definitions
in the model definition file for each module like:
```python
def custom_model():
    ...

# Adjust the learning rate according to iteration steps
def learning_rate_scheduler(model_version):
    if model_version < 5000:
        return 0.0003
    elif model_version < 12000:
        return 0.0002
    else:
        return 0.0001

# Process the prediction outputs for each batch
class PredictionOutputsProcessor(BasePredictionOutputsProcessor):
    ...
```
There will be multiple definition APIs for users to define different behaviors of model and the interfaces of APIs may be different. It is more convenient for users to define those behaviors using `tf.keras.callbacks.Callback`. 

Some use cases we observed that we want to support are:
* Case 1: Callback similar to `PredictionOutputsProcessor` that is executed after prediction outputs are made.
* Case 2: Callback to modulate learning rate as we are currently doing in `LearningRateScheduler`.
* Case 3: Callback to checkpoint model weights during training.
* Case 4: Callback to write additional summaries to TensorBoard after each evaluation job completes.
* Case 5: Callback to perform early stopping when certain conditions are met. For example, the metrics are met after an evaluation job.
* Case 6: Callback to upload model to remote storage after completing training.


To implement callbacks, we need to solve the following problems:
1. How to define callbacks for the model in ElasticDL.
2. How to call the methods of callbacks in ElasticDL.

## Define Callbacks for the Model in ElasticDL

In the model definition file, users can add `callbacks` API to define callbacks like:
```python
def custom_model():
    ...

def callbacks():
    early_stop_callback = EarlyStopCallback()
    learning_rate_scheduler = LearningRateScheduler()
    return [early_stop_callback, learning_rate_scheduler]


class EarlyStopCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        ...

class LearningRateScheduler(tf.keras.callbacks.Callback):
    def on_batch_start(self, batch, logs=None):
        ...
```

## Call the Methods of Callback in ElasticDL

We may define several callbacks for the model in a job. Tensorflow creates a container `CallbackList` to wrap the callbacks to conveniently call the methods in callbacks. We can call the methods in callbacks by calling the methods in `CallbackList`. For example:
```python
class CallbackList():
    def on_batch_end(self, batch, logs=None):
        for callback in self.callback_list:
            callback.on_batch_end(bach, logs)
```
For detail, we can view the [source code](https://github.com/tensorflow/tensorflow/blob/cf7fcf164c9846502b21cebb7d3d5ccf6cb626e8/tensorflow/python/keras/callbacks.py#L189-L196
) of `CallbackList` in Tensorflow.

So, we can also use CallbackList to wrap the callbacks in ElasticDL.
```python
from tensorflow.python.keras.callbacks import CallbackList
callbacks = callbacks()
callback_list = CallbackList(callbacks)
callback_list.on_batch_end(batch)
```


### Call the Methods of Callback using ParameterServerStrategy

Using ParameterServerStrategy in ElasticDL, we will launch instances as master, worker and parameter server. 
The master controls the start and end of job and epoch, the worker processes each batch and the parameter server update model weights with batch gradients. So we need to call different methods on different instance according the call frequency. Due to there are many methods in `tf.keras.callbacks.Callback` and we don't have use case for each method, so we only design to support calling methods which the use cases listed in the motivation section need. And we will design to support other methods if there is a use case.


### Call Methods of Callback Per Batch on Worker

The methods should be called per batch in `tf.keras.callbacks.Callback` are
```python
on_(train|test|predict)_batch_begin(self, batch, logs=None)
on_(train|test|predict)_batch_end(self, batch, logs=None)
```

"Case 1" needs to customize behavior by calling `on_predict_batch_end`. 
```python
def on_predict_batch_end(self, batch, logs=None):
    """Call on the prediction end of each batch
    Args:
        batch: integer, index of batch in the current worker
        logs: dict. Has keys batch, size and predictions representing
            the current batch number in the worker,
            the size of the batch,
            the prediction outputs of the batch.
    """
```

On the worker, we can call `on_predict_batch_end` after forward processing with batch features:
```python
def _process_minibatch(self, task_type, features, batch_index):
    ...
    if task_type == elasticdl_pb2.PREDICTION:
        predictions = self.forward_process(features)
        prediction_logs = {
            "batch": batch_index,
            "size": len(predictions)
            "predictions": predictions
        }
        self.callbacks_list.on_predict_batch_end(
            batch=batch_index, logs=prediction_logs
        )
```

### Call Methods of Callback Per Updating Batch Gradients on PS

"Case 2" and "Case 3" needs to customize behaviors After updating model weights using batch gradients on the PS. So, we add `on_update_batch_end` method, though it is not in `tf.keras.callbacks.Callback`.
```python
def on_update_batch_end(self, version, logs=None):
    """
    Args:
        version: integer, the model version in PS.
        logs: dict. Has keys batch and size representing the current batch number and the size of the batch.
    """
```
 
```python
def update_parameter_for_batch(gradients, model_version):
    update_gradient(gradients)
    for callback in callbacks_list.callbacks:
        if callback.hasattr("on_update_batch_end")
            callback.on_update_batch_end(self, batch=model_vesion)
```
However, we will implement a high performance parameter server using golang. And there is a question how golang can call the callbacks defined by users using Python.

### Call Methods of Callback Per Epoch on Master

The methods should be called per epoch in `tf.keras.callbacks.Callback` are
```python
on_epoch_begin(self, epoch, logs=None)
on_epoch_end(self, epoch, logs=None)
```

Only the master knows when an epoch begins and ends because the master creates tasks for each epoch using `_TaskDispatcher`. So, we can call those methods in the `get` and `report` of `_TaskDispatcher`. Due to there is no use case which needs to call `on_epoch_begin|end`, so we will design the support calling those methods in the future when we have the case.


### Call Methods of Callback When the Job Begins and Ends On Master

The methods should be called when the job begins and ends in `tf.keras.callbacks.Callback`
```python
on_(train|test|predict)_begin(self, logs=None)
on_(train|test|predict)_end(self, logs=None)
```

"Case 6" listed in the motivation section needs to customize behavior by calling `on_train_end` after a train job completes by calling `on_train_end`.
```python
def on_train_end(self, logs=None):
    """Call on the train job end
    Args:
        logs: dict. Currently no data is passed to this argument for this method but that may change in the future.
    """
```
The master determine whether or not the job is completed by `finished` method in `_TaskDispatcher`, so we can call `on_train_end` in the `finished` method if finished.
```python
def finished(self):
    """Return if all tasks are done"""
    finished = all([not self._todo, not self._eval_todo, not self._doing])
    if finished:
        self._callbacks_list.on_train_end()
    return finished
```

"Case 4" and "Case 5" listed in the motivation section need to customize behavior by calling `on_test_end` after a test job completes. 
```python
def on_test_end(self, logs):
    """Call on the test job end
    Args:
        logs: dict. Has key metrics representing the evaluation metrics on the validation data.
    """
```

The master determine whether or not the evaluation job is completed by `EvaluationService.complete_task()`. So, we can call `on_test_end` After `EvaluationService.complete_task()` returns evaluation metrics.
```python
if evaluation_task_completed:
    eval_metrics = self._evaluation_service.complete_task()
    if eval_metrics is not None:
        logs = {"metrics": eval_metrics}
        self._callbacks_list.on_test_end(logs)
```

### Call the Methods of Callback using AllReduceStrategy

The design to call the methods using AllReduceStrategy will be considered later.