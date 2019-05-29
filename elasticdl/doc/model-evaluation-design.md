## Model Evaluation Design

This document describes the design of model evaluation task for ElasticDL.

### Minimal Viable Product

#### Definitions

* `Model evaluation`: Computing metrics to judge the performance of the trained model.
* `Evaluator`: The pod responsible for performing model evaluation computations.
* `Multiprocessing`: Executing tasks in multiple threads in parallel on the same pod.

#### Requirements

* There's only one evaluator without multiprocessing.
* Master pod is responsible for creating the evaluator.
* Evaluator is created by master pod after training/task queue is finished.
* The evaluator fetches the latest model from master pod.
* Model can be evaluated by a specified number of steps. If `None`, evaluation will continue until reaching the end of input.
* Model evaluation metrics can be defined by users together with the model definition.
* The computed model evaluation metrics can be report back to master through RPC call.

#### Implementation Plan

* Reuse `WorkerManager` to create model evaluation pod.
* Implement `MasterServicer.ReportEvaluationMetrics()` and additional proto definitions such as
    `ReportEvaluationMetricsReply` and `ReportEvaluationMetricsRequest`.
* Implement `Evaluator` class that includes the following (reuse code in `Worker` whenever possible):
    * `distributed_evaluate()` that contains the main logic for model evaluation.
    * `report_task_result()` that reports evaluation task result back to master through RPC call.
    * `report_evaluation_metrics()` that reports the computed evaluation metrics back to master through RPC call.
* Add main CLI entry-point to `Evaluator.distributed_evaluate()`.

### Future Development

A list of potential features we may want for model evaluation in the future:

* Support evaluating the model during training, which includes:
    * `start_delay_secs`: Start evaluating after waiting for this many seconds.
    * `throttle_secs`: Do not re-evaluate unless the last evaluation was started at least this many seconds ago.
* `num_parallel_processes`: The number of children processes to run evaluation on each individual evaluator.
* `sample_weight`: Optional Numpy array of weights for the test samples, used for weighting the loss function.

### References

Some of the ideas are borrowed from existing solutions listed below:

* [`tf.keras.models.Model.evaluate()`](https://www.tensorflow.org/api_docs/python/tf/keras/models/Model#evaluate)
* [`tf.estimator.EvalSpec`](https://www.tensorflow.org/api_docs/python/tf/estimator/EvalSpec)
* [`tf.estimator.Estimator.evaluate()`](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#evaluate)
* [`tf.keras.metrics`](https://www.tensorflow.org/api_docs/python/tf/keras/metrics)