# Design to Support Custom Callback Using Keras API

This document describes the design for supporting callback to customize the behavior of model during training, evaluation and inference in ElasticDL.

## Motivation

In deep learning, we generally need to customize the behavior of model during training, evaluation or
inference, including reading/changing the model. We may perform the behavior per batch, per epoch or at the start and end of job. `tf.keras.callbacks.Callback` is an abstract base class and has methods to perform the behavior at different call frequency, such as `on_bath_end`, `on_epoch_end` and so on. So, we adopt the interfaces of `tf.keras.callbacks.Callback` to customize the behavior of model in ElasticDL. 

Now we have implemented some modules similar to callback in ElasticDL, such as `LearingRateScheduler` and `PredictionOutputsProcessor`. And users should write definitions in the model definition file for each module like:
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
There will be multiple definition APIs for users to define different behaviors of model. The interfaces of APIs may be different. It is more convenient for users to define those behaviors using `tf.keras.callbacks.Callback`. 

Some use cases we observed that we want to support are:
* Case 1: Callback similar to `PredictionOutputsProcessor` that is executed after prediction outputs are made.
* Case 2: Callback to modulate learning rate as we are currently doing in `LearningRateScheduler`.
* Case 3: Callback to write additional summaries to TensorBoard after each evaluation job completes.
* Case 4: Callback to perform early stopping when certain conditions are met. For example, the metrics are met after an evaluation job.
* Case 5: Callback to export model using SavedModel after the training is completed.
* Case 6: Callback to upload model to remote storage after the training is completed.

The following, we will design how to define and implement callbacks to support those cases.

## Define Callbacks for the Model in ElasticDL

In the model definition file, users can add `callbacks` API to define callbacks like:
```python
From elasticdl.python.callbacks import (
    PredictionOutputsProcessor,
    LearningRateScheduler
)

def custom_model():
    ...

def callbacks():
    prediction_output_processor = PredictionOutputsProcessor(process_fn)
    learning_rate_scheduler = LearningRateScheduler(schedule_fn)
    return [prediction_output_processor, learning_rate_scheduler]
```

## Initialize Callbacks and Set Callback Attributes in ElasticDL

### Use a Container for Callbacks Defined in the Model
We may define several callbacks for the model in a job. Tensorflow creates a container `CallbackList` to wrap the callbacks to conveniently call the methods in callbacks. For example:
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

### Set Default Attributes for Callbacks.
There are `set_model` and `set_params` in `tf.keras.callbacks.Callback`. `set_model` can set a model object to the attribute `model` of callback and `set_params` can set a dict object to the attribute `params` of callback.
We also can use those methods of `CallbackList` to set `model` and `params` for the callbacks in `CallbackList`. 
```python
model = custom_model()
callbacks = callbacks()
callback_list = CallbackList(callbacks)
callback_list.set_model(model)

params = {
    'batch_size': batch_size,
    'epochs': epochs,
    'saved_model_path': saved_model_path,
    'checkpoint_path': checkpoint_path
}
callback_list.set_params(params)
```

Then, we can call `model` and `params` in the methods like:
```python
class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        lr = self.model.optimizer.learning_rate,
        saved_model_path = self.params.get("save_model_path")
```


## Implement Callbacks to Support the Cases in the Motivation
We split the callbacks to two parts. One is the pre-configured callback which is automatically configured in ElasticDL and users don't need to define it in the model definition. And another is pre-made callback which users can configure in the model definition if needed.

### Pre-configured Callback in ElasticDL

* Case 5: Callback to export model using SavedModel after the training is completed.
```python
class SaveModelExporter(tf.keras.callbacks.Callback):
    """Export model using SavedModel after training
    Args:
        task_data_service: TaskDataService to process data according the task
        dataset_fn: function to process dataset
        model_handler: to transform the trained model with ElasticDL embedding layer to Keras native model.
    """
    def __init__(self, task_data_service, dataset_fn, model_handler):
        self._model_handler = model_handler
        self._task_data_service
        self._dataset_fn = dataset_fn

    def on_train_end(self, logs=None):
        """Call on the train job end
        Args:
            logs: dict. Currently no data is passed to this argument for this method but that may change in the future.
        """
        saved_model_path = self.params.get("saved_model_path", None)
        batch_size = self.params.get("batch_size")
        (task,dataset) = self._task_data_service.get_save_model_task_and_dataset()

        if task is not None and dataset is not None:
            dataset = self._dataset_fn(
                dataset,
                Mode.PREDICTION,
                self._task_data_service.data_reader.metadata,
            )
            dataset = dataset.batch(batch_size)
            model = self._model_handler.get_model_to_export(
                self.model, dataset
            )
            tf.saved_model.save(model, saved_model_path)
```

We have designed that the worker exports model using SavedModel in [Model Serving Design](https://github.com/sql-machine-learning/elasticdl/blob/develop/docs/designs/model_serving.md#export-the-model-with-elasticdllayersembedding-to-savedmodel) because the memory of the master may not be enough to load a model but the worker can. However, only the master knows when the training is completed. So, the master can create a training end task for the worker. The worker call `on_train_end` in callbacks after receiving the task. The task example is:
```python
train_end_callback_task = _Task(
    shard_name=shard_name,
    start=start_ind_this_task,
    end=end_ind_this_task,
    type=elasticdl_pb2.TRAIN_END_CALLBACK
)
```

### Pre-made Callbacks for Users to Configure in Model Definition
In the model definition, users can import pre-made callback class and configure them for the model. For example:
```python
from elasticdl.python.callbacks import (
    LearningRateScheduler,
    EarlyStopper
)

def callbacks():
    learning_rate_scheduler = LearningRateScheduler(schedule)
    early_stopper = EarlyStopper(stop_train)

def schedule(batch):
    return 0.003 if batch < 1000 else 0.001

def stop_train(metrics_list):
    latest_metrics = metrics_list[-1]
    return True if latest_metrics['auc'] > 0.8 else False
```

* Case 1: Callback to process prediction outputs after batch prediction outputs are made.
```python
def process_fn(predictions):
    """The function is defined by users
    Args:
        predictions: prediction outputs of the model
    """
    print(len(predictions))

class PredictionOutputsProcessor(tf.keras.callbacks.Callback):
    def __init__(self, process_fn):
        self.process_fn = process_fn
        super(PredictionOutputsProcessor, self).__init__()

    def on_predict_batch_end(self, batch, logs=None):
        """Call on the prediction end of each batch
        Args:
            batch: integer, index of batch in the current worker
            logs: dict. Has keys predictions representing
                the prediction outputs of a batch.
        """
        predictions = logs["predictions"]
        process(predictions)
```

* Case 2: Callback to modulate learning rate.
```python
def schedule_fn(version):
    """The function is defined by users
    Args:
        version: model iteration version
    """
    return 0.003 if batch < 1000 else 0.001

class LearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule_fn):
        super(LearningRateScheduler, self).__init__()
        self.schedule_fn = schedule_fn

    def on_train_batch_begin(self, batch, logs=None):
        """
        Args:
            batch: integer, the model version requested from PS.
            logs: dict. Has keys batch and size representing the current batch number and the size of the batch.
        """
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.schedule_fn(batch)
        K.set_value(self.model.optimizer.lr, lr)
```
Using ParameterServerStrategy, the worker calculates the batch gradients and set gradients to PS. PS updates weights using optimizer after receiving gradients. Although the worker can call `on_train_batch_begin` in `LearningRateScheduler` to adjust the learning rate of optimizer in its model, we should send the learning rate with gradients to PS by GRPC and PS updates weights using the received learning rate.

* Case 3: Callback to write additional summaries to TensorBoard after each evaluation job completes.
```python
class SummaryWriter(tf.keras.callbacks.Callback):
    def on_test_end(self, logs=None):
        """Call on the test job end
        Args:
            logs: dict. Has key metrics representing the evaluation metrics on the validation data.
        """
        metrics = logs.get("metrics", None)
        if metrics is None:
            return
        write(metrics)
```
The master determine whether or not the evaluation job is completed by `EvaluationService.complete_task()`. So, we can call `on_test_end` After `EvaluationService.complete_task()` returns evaluation metrics.
```python
if evaluation_task_completed:
    eval_metrics = self._evaluation_service.complete_task()
    if eval_metrics is not None:
        logs = {"metrics": eval_metrics}
        self._callbacks_list.on_test_end(logs)
```

* Case 4: Callback to perform early stopping when the metrics are met after an evaluation job.
```python
def stop_fn(metrics_list):
    """Function to determine whether or not to stop training
    Args:
        metrics_list: List to contain metrics of each evaluation job.
    Retrun:
        boolean: Stop training if true.
    """
    latest_metrics = metrics_list[-1]
    return True if latest_metrics['auc'] > 0.8 else False

class EarlyStopper(tf.keras.callbacks.Callback):
    def __init__(self, stop_fn):
        self.stop_train = stop_train
        self.metrics_list = []
        super(EarlyStopper, self).__init__()

    def on_test_end(self, logs=None):
        """Call on the test job end
        Args:
            logs: dict. Has key metrics representing the evaluation metrics on the validation data.
        """
        metrics = logs.get("metrics", None)
        if metrics is None:
            return
        self.metrics_list.append(metrics)
        self.model.stop = stop_train(self.metrics_list)
```
The same as `SummaryWriter`, the master call `on_test_end` of `EarlyStopper` After an evaluation job is completed.

* Case 6: Callback to upload model to remote storage after the training is completed.
```python
class ModelUploader(tf.keras.callbacks.Callback):
    """Upload model to remote storage
    """
    def __init__(self, remote_url):
        self.remote_url = remote_url
        super(ModelUploader, self).__init__()

    def on_train_end(self, logs=None):
        """Call on the train job end
        Args:
            logs: dict. Currently no data is passed to this argument for this method but that may change in the future.
        """
        saved_model_path = self.params["saved_model_path"]
        upload(save_model_path, remote_url)
```
The same as `SaveModelExporter`, the worker will call `on_train_end` of `ModelUploader` after receiving a train end task from the master.
