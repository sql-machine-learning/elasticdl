# Design to Support Keras Callback

This document describes the design for supporting callback to customize the behavior of model in ElasticDL.

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
* Callbacks used in testing that checks retry times and model parameter updates,
* Callback similar to `PredictionOutputsProcessor` that is executed after prediction outputs are made.
* Callback to write additional summaries to TensorBoard after each evaluation job completes.
* Callback to modulate learning rate as we are currently doing in `LearningRateScheduler`.
* Callback for logic to be executed before training starts such as starting embedding service.
* Callback to perform early stopping when certain conditions are met.
* Callback to upload model to remote storage after completing training.


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
The master controls the start and end of job and epoch. The worker and parameter server is responsible to process each batch. So we need to call different methods on different instance according the call frequency.


### Call Methods of Callback Per Batch

The methods should be called per batch in `tf.keras.callbacks.Callback` is
```python
on_(train|test|predict)_batch_begin(self, batch, logs=None)
on_(train|test|predict)_batch_end(self, batch, logs=None)
```
Each batch needs worker to calculate loss and gradient and parameter server to update gradients. So the worker and parameter server should call those methods per batch.

For worker:
```python
batch_index = 0
for batch in dataset:
    callback_list.on_(train|test|predict)_batch_begin(self, batch=batch_index, logs=None)
    process(batch)
    callback_list.on_(train|test|predict)_batch_end(self, batch=batch_index, logs=None)
    batch_index += 1
```

The parameter server only updates the gradients for each batch during training, so it doesn't need to 
call the methods per batch for test and prediction. 
```python
def update_parameter_for_batch(gradients, model_version):
    callback_list.on_train_begin(self, batch=model_version) # model_version is the iteration number during training.
    update_gradient(gradients)
    callback_list.on_train_end(self, batch=model_vesion)
```

### Call Methods of Callback Per Epoch

The methods should be called per epoch in `tf.keras.callbacks.Callback` is
```python
on_epoch_begin(self, epoch, logs=None)
on_epoch_end(self, epoch, logs=None)
```
Only the master knows when an epoch begins and ends because the master creates tasks for each epoch using `_TaskDispatcher`. So, we can call those methods in the `get` and `report` of `_TaskDispatcher`.
```python
def get(self, worker_id):
    if not self._todo and self._epoch < self._num_epochs - 1:
        # Start a new epoch
        self._epoch += 1
        self.create_tasks(elasticdl_pb2.TRAINING)
        logger.info("Starting epoch %d", self._epoch)
        self.callback_list.on_epoch_begin(self._epoch, logs)
        
        task = self._todo.pop()
        if not self._todo:
            self._epoch_end_task = task


def report(self, request, success):
    task_id = request.task_id
    if task_id == self._epoch_end_task and success:
        self.callback_list.on_epoch_end(self._epoch, logs)
```


### Call Methods of Callback When the Job Begins and Ends

The methods should be called when the job begins and ends in `tf.keras.callbacks.Callback`
```python
on_(train|test|predict)_begin(self, logs=None)
on_(train|test|predict)_end(self, logs=None)
```
The master starts and finishes a job, so it calls those methods. 

```python
def run(self):
    """
    The main loop of master.
    Dispatch the tasks to the workers until all the tasks are completed.
    """
    try:
        self.callback_list.on_(train|test|predict)_begin(self, logs=None)
        while True: 
            ...
    except KeyboardInterrupt:
        self.logger.warning("Server stopping")
    finally:
        self.callback_list.on_(train|test|predict)_end(self, logs=None)
        self._stop()
    return self._exit_code
```


### Call the Methods of Callback using AllReduceStrategy

The design to call the methods using AllReduceStrategy will be considered later.