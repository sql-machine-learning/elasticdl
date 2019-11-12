# Design for Allreduce-based Training Support

This document describes the design for supporting allreduce-based distributed training in ElasticDL.

## Motivation

While distributed training based on parameter servers can support training very large models and datasets by adding
more workers and parameter servers as described in [parameter server design doc](parameter_server.md), there are additional
challenges involved in order to optimize the performance, for example:

* It is not easy to identify the right ratio of the number of workers to the number of parameter servers. For example,
if only a small number of parameter servers are used, network communication will likely become the bottleneck for training.
If many parameter servers are used, the communication may saturate network interconnects.
* The memory quota of workers and parameter servers requires fine tuning to avoid out-of-memory or memory waste.
* If the model could fit within the computational resources on each worker, additional maintenance and communication overheads are
introduced when the model is partitioned to multiple parameter servers.
* We need to replicate the model on each parameter server in order to support fault-tolerance, which requires additional
computational and storage resources.

In contrast, distributed training based on collective communication primitives such as [allreduce](https://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/)
could be more efficient and easier to use in certain use cases. There are many existing technologies available that provide
implementations for these collective communication primitives and please head over to the section on
[existing collective communication technologies](#existing-collective-communication-technologies)
for details if interested. Allreduce-based distributed training could address many of the challenges mentioned above, for example:

* Each worker stores a complete set of model parameters. In other words, no parameter server is needed so it's straightforward
to add more workers when necessary.
* Failures among the workers can be recovered easily by restarting the failed workers and then load the current model from
any of the existing workers. Model does not need to be replicated to support fault-tolerance.
* The model can be updated more efficiently by fully leveraging the network structure and collective communication algorithms.
For example, in [ring-allreduce](http://research.baidu.com/bringing-hpc-techniques-deep-learning/) algorithm, each of *N* workers only needs to communicate with two of its peer workers *2 * (N − 1)* times
to update all the model parameters completely.
* Scaling up and down of the number of workers is as easy as reconstructing the underlying allreduce communicator and
re-assigning the ranks among the workers.

We'd like to support allreduce-based distributed training in ElasticDL so our users may obtain more efficient distributed
training performance in certain use cases without having to worry about many challenges involved in performance tuning and
resource allocation under parameter-server-based distributed training.

In the following sections, we will explain the design of allreduce-based distributed training in ElasticDL in detail.

For details on the existing technologies relevant to collective communications, please head over to the last section of
this design doc.

## Design Components

There are two major design components: the fault-tolerant allreduce implementation and support for allreduce-based training in ElasticDL.
The next two sections will illustrate the details of these components.

### Fault-tolerant Allreduce Implementation

We are collaborating with [Caicloud](https://github.com/caicloud/) on building an API that provides implementations of
fault-tolerant allreduce. The initial implementation will contain an experimental Python binding for NCCL that is
fault-tolerant and Kubernetes-native. This will include but not limited to the following objectives (more details to be disclosed later once
the implementation has been open-sourced):

* Fault-tolerant: if any of the worker pod fails, the [NCCL communicator](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/communicators.html)
can be reconstructed for the active worker pods. The allreduce operation continues as long as there's at least one healthy worker pod.
* Elastic: the number of worker pods can be dynamically added if there are enough computational resources available.
The NCCL communicator will be reconstructed and the ranks can be re-assigned as the number of worker pods changes.

The interface for allreduce would look like the following:

```python
num_epochs = 2
num_batches = 10
data_loader = DataLoader(num_batches)
communicator = AllreduceCommunicator()

for _ in range(num_epochs):
    for features, labels in data_loader:
        outputs = model(features)
        loss = loss_fn(outputs, labels)
        grads = calculate_gradients(loss, model)
        res, averaged_grads = communicator.average_gradients(grads)
        if res == SUCCESS:
            update_model(averaged_grads)
        elif res == FAILED:
            continue
```

We also provide API to broadcast all model parameters from one of the worker pods to all other active worker pods.
This would be useful to initialize the model parameters for a restarted or new joining worker pod. The interface would
look like the following:

```python
communicator = BroadcastCommunicator()
res, model_params = communicator.broadcast(from=worker_0_ip)
init_local_model(model_params)
```

There are a couple of implementation details that are worth mentioning here:

1. All NCCL API calls are asynchronous but our implementation of the communication collective primitives is synchronous
so it's easier for other frameworks like ElasticDL to use without worrying about failure handling.
1. It is not necessary to pass the list of IPs of the worker pods to either `AllreduceCommunicator` or `BroadcastCommunicator`
since the shared storage used in the implementation would keep track of the list of IPs.

### Allreduce-based Training in ElasticDL

#### Gradients Averaging and Model Updating

For training based on parameter servers, gradients calculation and model updating include the following steps:

1. Send the gradients calculated locally from each worker to parameter servers.
1. Calculate the average of all the received gradients on master.
1. Update the model in parameter servers.

On contrary, in allreduce-based training, each worker in ElasticDL calculates gradients locally and then calculates
the average of gradients across all workers using collective communication via `AllreduceCommunicator.average_gradients()`
that we mentioned in the previous section. The main differences are the following:

1. Gradients from each worker are not sent to master.
1. The average of gradients across all workers is calculated locally on each worker.
1. The model is updated directly on each worker and each worker has the exact same copy of the model.

Below is the pseudo-code for this process on each worker:

```python
communicator = AllreduceCommunicator()

with tf.GradientTape() as tape:
    outputs = model(features, training=True)
    loss = loss_fn(outputs, labels)
local_grads = tape.gradient(loss, get_trainable_items())
res, averaged_grads = communicator.average_gradients(grads)
if res == SUCCESS:
    update_model(averaged_grads)
else:
    report_failure()
    continue
```

#### Failure Handling during Training

The above pseudo-code will be wrapped and executed for each batch of the dataset inside `process_minibatch_and_report()`
in the code below. Each worker continues to perform tasks until there is no new batch available.

```python
while True:
    dataset = self._task_data_service.get_dataset()
    if not dataset:
        break
    dataset = dataset.batch(self._minibatch_size).prefetch(1)
    for dataset_batch in dataset:
        task = self._task_data_service.get_current_task()
        process_minibatch_and_report(dataset_batch, task)
```

If any of the gradients averaging operation fails while the workers are still active and healthy, we simply report the
failure and continue training on the next batch.

If any of the workers fails, e.g. the pod is accidentally killed, the following steps will be performed:

1. The task that the failed worker was handling will be added back to the task queue.
1. Since the `AllreduceCommunicator` is aware of the failure. It will reconstruct the communicator and re-assign ranks among
the existing active workers. The existing workers will then continue to run on the tasks at hand.
1. The master pod will try to create a new worker pod after a specified restart delay period.
1. Once the worker pod becomes active and `AllreduceCommunicator` is aware of it, we perform the following steps:
    1. Lock all existing worker pods.
    1. Master pod selects one of the worker pods to broadcast the current model parameters to all worker pods.
    1. Release the lock that each worker pod holds.
    1. All worker pods continue to process training tasks at hand and perform allreduce operations.

Note that since the existing active workers have the exact same copy of the model after the previous allreduce operation completes,
we can guarantee that the new worker will have the same copy of the model as the ones on other workers once the next
allreduce operation completes.

#### Training with Evaluation

If the worker encounters any evaluation tasks in the above process, it will evaluate the model directly on the worker once the
under-going allreduce-based gradients averaging has completed and the model has been updated. The behavior is the same as
what's described in [model evaluation design doc](model_evaluation.md) except that we are evaluating the model on workers
instead of on parameter servers. Once an evaluation completes, we send the evaluation result to master for TensorBoard
service to consume.

#### ElasticDL Embedding Layer

ElasticDL embedding layer is not supported under allreduce-based training since each worker has the exact same copy of
the model that must fit within each worker's specified resources. Any layers in the model defined via
``elasticdl.layers.embedding`` will be replaced by `tf.keras.layers.Embedding`.

#### Relevant CLI Arguments

The following CLI arguments are relevant and their behaviors might be changed under allreduce-based training:

* ``--restart_policy``: The pod restart policy when pod crashed.
the `AllreduceCommunicator` just reconstructed the communicator and we want to wait for a while before restarting the
failed worker pod which requires reconstruction of the communicator again once the pod becomes active. 
* ``--distribution_strategy``: In addition to the existing "ParameterServerStrategy" that we have, we add a new strategy
called "AllreduceStrategy".
* ``--num_ps_pods`` will be ignored if "AllreduceStrategy" is used and only ``--num_workers`` will be taken into account.
* ``--use_async`` and ``--lr_staleness_modulation`` will be ignored if "AllreduceStrategy" is used.
* Only ``--grads_to_wait = 1`` will be supported if "AllreduceStrategy" is used.

The following new CLI arguments are added:

* ``--restart_delay_secs``: The number of seconds to delay before restarting the failed pods. This could be useful when


## Potential Future Optimizations

* We can potentially overlap the backward computations and gradient optimizations. More discussions on this can be found
in [this Github issue](https://github.com/tensorflow/tensorflow/issues/33274).
* For models with a large amount of tensors, such as ResNet, many small allreduce operations are needed. In this case,
we could fuse multiple small tensors together before performing allreduce operations to maximize performance since
allreduce utilizes the network in an optimal way if the tensors are large enough.
* Since the gradients of the embedding layer are row-sparse, supporting sparse allreduce for embedding layers
will decrease communication costs.
* Since the number of workers may change during training, the batch size (sum of the size of minibatches on each worker)
will affect the model training accuracy. We can support customized learning rate scheduler, which will take epoch/batch_size
into account. We can also support [LARS (Layer-wise Adaptive Rate Scaling)](https://arxiv.org/abs/1708.03888) so that
large batch size can be used.

## Existing Collective Communication Technologies

The following three libraries provide collective communications and all of them have been adopted by large projects:

* [MPI](https://www.mpi-forum.org/)
* [NCCL](https://github.com/NVIDIA/nccl)
* [Gloo](https://github.com/facebookincubator/gloo/)
* [Rabit](https://github.com/dmlc/rabit)

### MPI

Message Passing Interface (MPI) is a standardized and portable message-passing standard designed by a group of researchers
from academia and industry to function on a wide variety of parallel computing architectures.

There are several well-tested and efficient implementations of MPI, such as [MPICH](https://www.mpich.org/about/overview/)
and [Open MPI](https://www.open-mpi.org/). Some recent implementations, such as [MVAPICH](https://developer.nvidia.com/mvapich)
and [IBM Spectrum MPI](https://developer.nvidia.com/ibm-spectrum-mpi), are also able to take advantage of CUDA IPC and GPU Direct technologies in order to avoid memory copies through the CPU.
However, these implementations of MPI do not support fault tolerance.

### NCCL

NVIDIA Collective Communications Library (NCCL) is a library of multi-GPU collective communication primitives that are topology-aware and
can be easily integrated into applications. It has been adopted by both [TensorFlow](https://github.com/tensorflow/tensorflow/) and [PyTorch](https://github.com/pytorch/pytorch).

NCCL focuses on accelerating collective communication primitives. For example, NCCL conveniently removes
the need for developers to optimize their applications for specific machines. In addition, NCCL provides fast collectives
over multiple GPUs both within and across nodes. It supports a variety of interconnect technologies including PCIe, NVLINK,
InfiniBand Verbs, and IP sockets. NCCL also automatically patterns its communication strategy to match the system’s underlying
GPU interconnect topology.

NCCL does not support fault tolerance but one can support this by filtering out the failed workers, reassigning ranks, and then
reconstructuring the [NCCL Communicator](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/communicators.html).
Also note that NCCL only supports GPUs for the collective communication primitives.

### Gloo

Gloo is a collective communications library. It comes with a number of collective algorithms useful for machine learning
applications, which includes but not limited to broadcast and allreduce. It has been adopted by [PyTorch](https://github.com/pytorch/pytorch).

Transport of data between participating machines is abstracted so that IP can be used at all times, or InifiniBand (or RoCE)
when available. In the latter case, if the InfiniBand transport is used, GPUDirect can be used to accelerate cross machine
GPU-to-GPU memory transfers. Gloo includes several collective algorithm implementations that work directly with NVIDIA GPU buffers.
These take advantage of overlapping host and GPU operations to decrease overall latency.

Gloo does not support fault tolerance but supports both GPUs and CPUs for at least allreduce and broadcast primitives.
The implementation of the collective operations for CUDA tensors is not as optimized as the ones provided by the NCCL backend.

### Rabit

Rabit is a lightweight library that provides a fault tolerant interface of allreduce and broadcast. It has been adopted
by [XGBoost](https://github.com/dmlc/xgboost) and [Apache MXNet](https://github.com/apache/incubator-mxnet).

Rabit provides fault tolerance via the following steps:

* If a worker fails, other workers will pause before the failed worker recovers
* Once the failed worker restarts, load the latest checkpoint from one of the existing workers and continue running

Since all the workers will get the same result after calling allreduce/broadcast. Any of the workers can record the history
of allreduce/broadcast call results. The restarted node can be recovered correctly and continue running with existing workers.
More details on this can be found [here](https://rabit.readthedocs.io/en/latest/guide.html#fault-tolerance).

A couple of things worth mentioning are:

* The checkpoints are saved to memory instead of disk
* All the alive workers will be blocked in subsequent allreduce calls

Rabit assumes the number of workers is fixed so if somehow the failed worker cannot be recovered, e.g. due to lack of
resources, then the whole Rabit process will be stuck. The network topology that Rabit constructs can only be recovered
instead of being modified based on the number of available workers. In other words, the fault tolerance of Rabit cannot
support elastic scheduling.

Rabit supports many networking options through its [MPI support](https://github.com/dmlc/rabit/blob/master/src/engine_mpi.cc)
which is not fault tolerant given that the implementation is based on MPI. If fault tolerance is enabled through Rabit's [robust implementation](https://github.com/dmlc/rabit/blob/master/src/allreduce_robust.cc),
Rabit only supports TCP networking but not others like RDMA and InfiniBand. Though it provides an interface
so developers can write implementations based on other frameworks such as NCCL and Gloo that provide additional networking
options. There's an ongoing work for Gloo implementation in Rabit [here](https://github.com/dmlc/rabit/pull/113).
