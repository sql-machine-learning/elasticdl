import threading


class LearningRateScheduler:
    def __init__(self, learning_rate=0.001, model_version=0):
        """Constructs a `LearningRateScheduler` instance.
        Args:
            learning_rate: The learning rate to be modulated.
                This can be either a numeric value or a callable.
                If callable, it takes model_version as input.
            model_version: the initial model version
        """
        self._learning_rate = learning_rate
        self._tls = threading.local()
        self._tls.model_version = model_version

    def set_model_version(self, model_version):
        """Sets the model version
        Args:
            model_version: the model version to set
        """
        self._tls.model_version = model_version

    def get_learning_rate(self):
        """Gets the current learning rate accordingn to scheduler..
        Returns:
            The learning rate modulated by the multiplier.
        """
        lr = self._learning_rate
        if callable(lr):
            lr = lr(self._tls.model_version)
        return lr


def add_lr_scheduler_to_optimizer(optimizer, lr_scheduler):
    """Adds learning rate scheduler to the given optimizer.
    Args:
      optimizer: The optimizer to add learning rate scheduler to.
      lr_scheduler: learning rate scheduler
    Returns:
      A `LearningRateScheduler` instance.
    """
    # Replace the learning rate in optimizer with a callable
    lr_scheduler = LearningRateScheduler(lr_scheduler)
    optimizer.learning_rate = lr_scheduler.get_learning_rate

    return lr_scheduler
