# Design Doc: Asynchronous SGD (Stochastic Gradient Descent)

## Motivation

In the paper [Large scale distributed deep networks][Dean], the authors describle two kinds of distributed training:

+ Model parallelism:
  + A Large model is splited and placed on multiple workers. Each worker is responbile for its own part of model weights. They train the model
  with the same data.
+ Data parallelism
  + Each worker has a complete model replica and they share the same parameter servers. They train the model with different data shards.

In ElasticDL, we implement data parallelism. Multiple workers get data tasks from master. After processing the task, workers report
their gradients and a local model version to parameter servers. Parameter servers collect `grad_to_wait`
gradients from workers before updating the model (averaging the gradients and applying the averaged gradient).
It tracks the model updates with a variable `_model_version`. Whenever parameter servers update the model, they also update `_model_version`.
For consistency, both of previous updating operations are guarded with a `threading.Lock` object.

The training process in ElasticDL is synchronous:

+ Whenever any worker reports gradients to parameter servers, together it reports its local model version. If this version is not equal to parameter servers'
model vesion, the gradients will be rejected. The rejected worker will retry by getting new model, calculating gradients and reporting them.
+ Whenever parameter servers receive gradients, they have to acquire the lock before gathering or updating the model.
+ Whenever a worker sends `GetModel` request, parameter servers have to acquire the lock.

The synchronization comes at the cost of wasted compuation of workers and blocked computation of parameter servers. The cost will hurt the traning speed. As data volume increases dramatically nowdays, a higher training speed is essential for the application of a model. Asynchronous SGD will mitigate those defects by allowing
for asynchronously updating the model without any lock.

### Design

In asynchronous SGD, workers report gradients and pull new model without waiting for parameter servers' acknowledgement. Parameter servers apply gradients
update immediately instead of gathering `grads_to_wait` gradients and averaging them with the lock object.

Asynchronous SGD will introduce staleness in parameters. The staleness means:

+ Workers may get parameters from different step.
+ Parameter servers may apply gradients update that are not based on current parameters stored.

Due to the staleness of parameters, asynchronous SGD takes a longer time to converge and the model quality is not as good as synchronous SGD sometimes.
In order to reach a balance between synchronous SGD and asynchronous SGD, we adopt the strategy of SSP (Stale synchronous parallel) from the paper
[More effective distributed ML via a stale synchronous parallel parameter server][EricXing]. It controls the maximum step difference of the fastest and slowest worker.

We introduce two classes to help implement SSP strategy.

+ `StalenessStrategy`
+ `GradientUpdatingStrategy`

#### `StalenessStrategy`

`StalenessStrategy` decides whether a worker needs to pull new model parameters before processing next data batch. It makes its decision according to:
+ Worker's local model version
+ Parameter servers' model version
+ Current fastest worker's model version
+ Current slowest worker's model version

#### `GradientUpdatingStrategy`

`GradientUpdatingStrategy` controls how parameter servers apply graidents update.

+ `DirectUpdatingStrategy`: directly updates the model.
+ `AverageUpdatingStrategy`: gathers `grads_to_wait` graidents, averages them and apply the averaged gradient update.

By combining `StalenessStrategy` and `GradientUpdatingStrategy`, we can also achieve synchronous SGD.

## Reference

+ [Revisiting distributed synchronous SGD][JianminChen]
+ [Large scale distributed deep networks][Dean]
+ [More effective distributed ML via a stale synchronous parallel parameter server][EricXing]

[Dean]: http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf
[EricXing]: http://www.cs.cmu.edu/~seunghak/SSPTable_NIPS2013.pdf
[JianminChen]: https://arxiv.org/abs/1604.00981
