from abc import ABC, abstractmethod


class BasePredictionOutputsProcessor(ABC):
    """
    This is the base processor for prediction outputs.
    Users need to implement the abstract methods in order
    to process the prediction outputs.
    """

    @abstractmethod
    def process(self, predictions, worker_id):
        """
        The method that uses to process the prediction outputs produced
        from a single worker.

        Arguments:
            predictions: The raw prediction outputs from the model.
            worker_id: The ID of the worker that produces this
                batch of predictions.
        """
        pass
