import tensorflow as tf

from elasticdl.proto import elasticdl_pb2


class MaxStepsStopping(tf.keras.callbacks.Callback):
    """Stop training if the training steps exceed the maximum.

    Args:
        max_steps:

    Example:
    ```python
    from elasticdl.python.elasticdl.callbacks.max_steps_stopping import (
        MaxStepsStopping
    )

    def callbacks():
        # This callback will stop the training when the training steps
        # exceed the max_steps.
        max_steps_stopping = MaxStepsStopping(max_steps=1000)
        return [max_steps_stopping]
    ```
    """

    def __init__(self, max_steps):
        self._max_steps = max_steps
        self._completed_steps = 0

    def on_task_end(self, task, logs=None):
        """Call on the task end
        Args:
            task: A completed task.
            logs: dict. Currently no data is passed to this argument for this
                method but that may change in the future.
        """
        batch_size = self.params.get("batch_size", None)
        if task.type == elasticdl_pb2.TRAINING:
            task_records = task.end - task.start
            task_batch_count = int(task_records / batch_size)
            self._completed_steps += task_batch_count
            if self._completed_steps > self._max_steps:
                self.model.stop_training = True
