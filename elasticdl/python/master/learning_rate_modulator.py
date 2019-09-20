import threading


class LearningRateModulator:
    """Modulates the learning rate with a multiplier.

    Note:
        This class supports concurrent usage by using
        thread local storage.
    """

    def __init__(self, learning_rate):
        """Constructs a `LearningRateModulator` instance.

        Args:
            learning_rate: The learning rate to be modulated.
                This can be either a numeric value or a callable.
        """
        self._learning_rate = learning_rate
        self._tls = threading.local()
        self._tls.multiplier = 1

    def set_multiplier(self, multiplier):
        """Sets the multiplier.

        Args:
            multiplier: The multiplier used to modulate the learning rate.
        """
        self._tls.multiplier = multiplier

    def get_learning_rate(self):
        """Gets the modulated learning rate.

        Returns:
            The learning rate modulated by the multiplier.
        """
        lr = self._learning_rate
        if callable(lr):
            lr = lr()
        lr *= self._tls.multiplier
        return lr


def add_lr_modulation_to_optimizer(optimizer):
    """Adds learning rate modulation to the given optimizer.

    Args:
      optimizer: The optimizer to add learning rate modulation to.

    Returns:
      A `LearningRateModulator` instance.
    """
    # Get learning rate from optimizer
    learning_rate = optimizer._hyper["learning_rate"]

    # Replace the learning rate in optimizer with a callable
    lr_modulation = LearningRateModulator(learning_rate)
    optimizer.learning_rate = lr_modulation.get_learning_rate

    return lr_modulation
