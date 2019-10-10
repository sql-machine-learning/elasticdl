## Model Evaluation Design

This document describes the design of model evaluation task for ElasticDL.

### Minimal Viable Product

#### Definitions

* `Model evaluation`: Computing metrics to judge the performance of the trained model.
* `Evaluation worker`: The worker responsible for performing model evaluation task.
* `Multiprocessing`: Executing tasks in multiple threads in parallel on the same pod.

#### Requirements

* There's only one evaluation worker without multiprocessing.
* Master pod is responsible for creating the evaluation worker.
* Evaluation worker is created by master pod together with the workers for training.
* Evaluation starts after a specified warm-up period and on a given time interval. For example, we need to expose
    the following parameters to users:
    * `start_delay_secs`: Start evaluating after waiting for this many seconds.
    * `throttle_secs`: Do not re-evaluate unless the last evaluation was started at least this many seconds ago.
* The evaluation worker fetches the latest model from master pod.
* Model can be evaluated by a specified number of steps or batches of evaluation samples. If `None`,
    evaluation will continue until reaching the end of input.
* Model evaluation metrics can be defined by users together with the model definition.
* The computed model evaluation metrics can be report back to master through RPC call.

#### Implementation Plan

* Implement `MasterServicer.ReportEvaluationMetrics()` and additional proto definitions such as
    `ReportEvaluationMetricsReply` and `ReportEvaluationMetricsRequest`.
* Extend `Worker` to support the following:
    * `distributed_evaluate()` that contains the main logic for model evaluation.
    * `report_task_result()` that reports evaluation task result (e.g. task id and error message) back to master through RPC call.
    * `report_evaluation_metrics()` that reports the computed evaluation metrics (e.g. accuracy, precision, recall, etc.) back to master through RPC call.
* Add main CLI entry-point to `Worker.distributed_evaluate()` that will be used in `WorkerManager`.
* Extend `WorkerManager` to support the following:
    * Instantiate a separate evaluation task queue from evaluation data directory.
    * Start an evaluation worker from evaluation task queue.
    * Update `master.main()` to support model evaluation task if user requested.

### Future Development

A list of potential features we may want for model evaluation in the future:

* `num_parallel_processes`: The number of children processes to run evaluation on each individual evaluation worker.
* `sample_weights`: Optional Numpy array of weights for the test samples, used for weighting the loss function.

### References

Some of the ideas are borrowed from existing solutions listed below:

* [`tf.keras.models.Model.evaluate()`](https://www.tensorflow.org/api_docs/python/tf/keras/models/Model#evaluate)
* [`tf.keras.metrics`](https://www.tensorflow.org/api_docs/python/tf/keras/metrics)
* [`tf.estimator.EvalSpec`](https://www.tensorflow.org/api_docs/python/tf/estimator/EvalSpec)
* [`tf.estimator.Estimator.evaluate()`](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#evaluate)
* [`tf.estimator.train_and_evaluate()`](https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate)
