import tensorflow as tf

from elasticdl.python.common.constants import Mode


class SavedModelExporter(tf.keras.callbacks.Callback):
    """Export model using SavedModel after training.
    Args:
        task_data_service: TaskDataService to process data according the task
        dataset_fn: function to process dataset
        model_handler: to transform the trained model with ElasticDL embedding
            layer to Keras native model.
    """

    def __init__(self, task_data_service, dataset_fn, model_handler):
        self._model_handler = model_handler
        self._task_data_service
        self._dataset_fn = dataset_fn

    def on_train_end(self, logs=None):
        """Call on the train job end
        Args:
            logs: dict. Currently no data is passed to this argument for this
                method but that may change in the future.
        """
        saved_model_path = self.params.get("saved_model_path", None)
        batch_size = self.params.get("batch_size")
        task = self._task_data_service.get_train_end_callback_task()
        dataset = self._task_data_service.get_dataset_by_task(task)
        if dataset is not None:
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
