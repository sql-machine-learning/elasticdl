import threading


class LearningRateModulation:
    """
    Modify learning rate with a multiplier.
    Support concurrent usage by using thread local storage.
    Arguments
      learning rate: can be a value or a callable.
    """

    def __init__(self, learning_rate):
        self._learning_rate = learning_rate
        self._tls = threading.local()
        self._tls.multiplier = 1

    def set_multiplier(self, multiplier):
        self._tls.multiplier = multiplier

    def get_learning_rate(self):
        lr = self._learning_rate
        if callable(lr):
            lr = lr()
        lr *= self._tls.multiplier
        return lr


def add_lr_modulation_to_optimizer(optimizer):
    """
    Add lr modulation feature in optimizer
    Argument:
      optimizer: the optimizer to add lr modulation feature
    Return:
      LearningRateModulation instance
    """
    # Get learning rate from optimizer
    learning_rate = optimizer._hyper["learning_rate"]

    # Replace the learning rate in optimizer with a calllable
    lr_modulation = LearningRateModulation(learning_rate)
    optimizer.learning_rate = lr_modulation.get_learning_rate

    return lr_modulation
