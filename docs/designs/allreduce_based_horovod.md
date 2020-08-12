# Design for Horovod Based AllReduce

This document describes the design for supporting AllReduce-based distributed
training based on [Horovod](https://github.com/horovod/horovod) in ElasticDL.

## Motivation

We have developed elastic AllReduce based on FTlib in ElasticDL.
From the [benchmark report](../benchmark/ftlib_benchmark.md), we can
find that the performance of FTlib for ResNet50 is worse than Horovod.

FTlib uses the gossip protocol to build consensus, which is not stable
as described in this [issue](https://github.com/sql-machine-learning/elasticdl/issues/2192#issuecomment-664096185).

FTlib uses Gloo to implement elastic AllReduce because the worker process
can catch the exception from Gloo and not exit if any collective communications
fail. Horovod also can use Gloo as the backend.
There are many small parameter tensors in the ResNet50 model. We have to
launch an AllReduce operator to synchronize each tensor. It brings a lot
of overhead. There are many optimizations like [Tensor Fusion](https://horovod.readthedocs.io/en/latest/tensor-fusion_include.html)
in Horovod to reduce the overhead. So, the performance of Horovod for ResNet50
is better than FTlib.

On the Kubernetes cluster, we usually use [kubeflow/mpi-operator](https://github.com/kubeflow/mpi-operator)
to submit a Horovod job. kubeflow/mpi-operator is not fault-tolerant and
elastic. Horovod has supported [elastic training](https://horovod.readthedocs.io/en/latest/elastic_include.html)
which can scale up and down the number of workers dynamically at runtime.
Elastic Horovod needs a shell script to discover worker hosts on the cluster.
However, it is difficult for users to use Kubernetes API to discover
worker pod hosts. What's more, data access is a problem for data parallel
training it the number of workers changes. Random sampling is a solution
that may affect the training accuracy. There is a master process in ElasticDL.
The master can get all worker hosts by Kubernetes API and dynamically
assign data shards for workers to solve data access for elastic training.
So, it is more user-friendly to run an elastic AllReduce-based training
job using ElasticDL with Horovod.

## ElasticDL Re-initialize Horovod When the Number of Workers Changes

When using Horovod with Gloo backend, we need to create a `RendezvousServer`,
which has a KVStore. Also, we need to put a host plan into this KVStore.
A host plan includes worker hosts and their assigned ranks,
which are required by Gloo. The master in ElasticDL is responsible for creating
the RendezvousServer. To support elastic training, when the master detects
the number of workers changes, it will create a new host plan and put it
in the KVStore.

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

When the master finds the number of workers changes, it can re-create a new
`RendezvousServer` and notify workers to re-initialize Horovod.
In the Kubernetes cluster, the number of workers may change for the
following reasons:

1. Some workers fail because of preemption.
1. A worker pod status becomes running.

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
