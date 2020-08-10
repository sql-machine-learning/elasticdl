# Design for Elastic AllReduce-based Training Support based on Horovod

This document describes the design for supporting AllReduce-based distributed
training based on Horovod in ElasticDL.

## Motivation

We have developed elastic AllReduce based on FTlib in ElasticDL.
From the [benchmark report](../benchmark/ftlib_benchmark.md), we can
find that the performance of FTlib is worse than Horovod with the
same backend Gloo. Because there are many optimizations like
[Tensor Fusion](https://horovod.readthedocs.io/en/latest/tensor-fusion_include.html)
in Horovod to improve performance. When we use Horovod with Gloo backend,
the training process cannot crash when the number of workers
changes. The workers can rebuild consensus to continue training.
So, we can use Horovod to implement elastic AllReduce-based Training
in ElasticDL.

What's more, it is not stable to build consensus by gossip protocol in FTlib
like the [issue](https://github.com/sql-machine-learning/elasticdl/issues/2192#issuecomment-664096185).
In ElasticDL, we can utilize master to notify workers to build
the consensus when the number of workers changes.

So, the design describes how to use ElasticDL master to build a consensus of workers
and use Horovod with Gloo backend to implement elastic AllReduce.

## Initialize the Horovod Context with Gloo Backend

A Horovod job contains a driver and workers. The driver does not participate
in the model training and is responsible for setting worker hosts.
When we use Horovod with Gloo, the steps to initialize Gloo context are:

1. The driver gets worker hosts and slots from the start command or a discovery
script.
1. The driver creates a `RendezvousServer` and set worker hosts into the KVStore
implemented by HTTP for Gloo.
1. The driver starts worker processes to execute the training loop.
1. The worker calls `hvd.init` initialize the Gloo context for AllReduce.

If the worker hosts change, we can re-create a `RendezvousServer` and reset
worker hosts into the KVStore. Then the worker calls `hvd.shutdown` and `hvd.init`
to re-initialize the Gloo context.

## ElasticDL Re-initialize the Horovod Context When the Worker Number changes

In ElasticDL, there is a master to manage workers. The master can get all worker
hosts and detect whether the number of workers changes. So, we can use the master to create a
`RendezvousServer` similar to the driver in Horovod. After the master launches
workers, it can set worker hosts into KVStore of Horovod for Gloo.

```python
import horovod
from horovod.run.http.http_server import  RendezvousServer
from horovod.runner.common.util.hosts import get_host_assignments

hosts = get_worker_hosts()

host_alloc_plan = get_host_assignments(hosts, num_proc)
global_rendezv_port = rendezvous.start()

# Set hosts into KVStore for Gloo
rendezvous.init(host_alloc_plan)
```

Then, the worker can call `hvd.init` to initialize the Gloo context for
AllReduce.

When the master finds the worker number changes, it can re-create a new
`RendezvousServer` and notify workers to re-initialize the Gloo Context.
In the Kubernetes cluster, the ElasticDL worker number may change for the
following reasons:

1. Some workers fail because of preemption.
1. The master re-launches new workers.

In the first case, the Horovod AllReduce operator will raise an exception
and the worker can catch the exception and re-initialize.

In the second case, the worker will query the master periodically to see
if there are new workers and re-initialization of the AllReduce process
the group is needed.

## The Worker Averages Gradients Using Horovod

Using TensorFlow eager execution, we can use `hvd.DistributedGradientTape`
to wrap `tf.GradientTape` to average gradients.

```python
@tf.function
def training_process_with_horovod(self, features, labels):
    with tf.GradientTape() as tape:
        outputs = self._model.call(features, training=True)
        loss = self._loss(labels, outputs)
    tape = hvd.DistributedGradientTape(tape)
    grads = tape.gradient(loss, mnist_model.trainable_variables)
    return loss, grads
```

If some workers fail, the `hvd.DistributedGradientTape` will raise
a `tensorflow.python.framework.errors_impl.UnknownError`. We can catch
the error and re-initialize the Horovod context if the error contains
`HorovodAllreduce`, `HorovodAllgather`, or `HorovodBroadcast`.

```python
def training_process_horovod_fault_tolerance(self, freature, labels)
    from tensorflow.python.framework.errors_impl import UnknownError
    initialize_horovod = False

    hosts_update = query_worker_hosts_updated(master)
    if hosts_updated:
        initialize_horovod = True

    if not initialize_horovod:
        try:
            loss, grads = self.training_process_with_horovod(features, labels)
        except UnknownError as e:
            if ('HorovodAllreduce' in e.message or
                'HorovodAllgather' in e.message or
                'HorovodBroadcast' in e.message):
                initialize_horovod = True

    if initialize_horovod:
        hvd.shutdown()
        hvd.init()
```

After initializing Horovod, we should broadcast the model in alive workers to
the new workers. The master can assign rank 0 to the oldest worker, as it will
be used as the broadcast source to synchronize models among workers.

```python
from horovod.tensorflow.functions import broadcast_variables
def _broadcast_model(model, optimizer, backend):
    broadcast_variables(model.variables, root_rank=0)
    broadcast_variables(optimizer.variables(), root_rank=0)
```
