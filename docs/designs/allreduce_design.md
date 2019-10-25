# Design for Allreduce Support

This document describes the design for supporting Allreduce in ElasticDL. Note that this is still a work-in-progress.

## Existing Collective Communication Technologies

The following three libraries provide collective communications and all of them have been adopted by large projects:

* [MPI](https://www.mpi-forum.org/)
* [Gloo](https://github.com/facebookincubator/gloo/)
* [NCCL](https://github.com/NVIDIA/nccl)
* [Rabit](https://github.com/dmlc/rabit)

### MPI

Message Passing Interface (MPI) is a standardized and portable message-passing standard designed by a group of researchers
from academia and industry to function on a wide variety of parallel computing architectures.

There are several well-tested and efficient implementations of MPI, such as [MPICH](https://www.mpich.org/about/overview/)
and [Open MPI](https://www.open-mpi.org/). However, these implementations of MPI do not support fault tolerance.

### Gloo

Gloo is a collective communications library. It comes with a number of collective algorithms useful for machine learning
applications, which includes but not limited to Broadcast and Allreduce. It has been adopted by [PyTorch](https://github.com/pytorch/pytorch).

Transport of data between participating machines is abstracted so that IP can be used at all times, or InifiniBand (or RoCE)
when available. In the latter case, if the InfiniBand transport is used, GPUDirect can be used to accelerate cross machine
GPU-to-GPU memory transfers. Gloo includes several collective algorithm implementations that work directly with NVIDIA GPU buffers.
These take advantage of overlapping host and GPU operations to decrease overall latency.

Gloo does not support fault tolerance but supports both GPUs and CPUs for at least Allreduce and Broadcast primitives.

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

### Rabit

Rabit is a lightweight library that provides a fault tolerant interface of Allreduce and Broadcast. It has been adopted
by [XGBoost](https://github.com/dmlc/xgboost) and [Apache MXNet](https://github.com/apache/incubator-mxnet).

Rabit provides fault tolerance via the following steps:

* If a worker fails, other workers will pause before the failed worker recovers
* Once the failed worker restarts, load the latest checkpoint from one of the existing workers and continue running

Since all the workers will get the same result after calling Allreduce/Broadcast. Any of the workers can record the history 
of Allreduce/Broadcast call results. The restarted node can be recovered correctly and continue running with existing workers.
More details on this can be found [here](https://rabit.readthedocs.io/en/latest/guide.html#fault-tolerance).

A couple of things worth mentioning are:

* The checkpoints are saved to memory instead of disk
* All the alive workers will be blocked in subsequent Allreduce calls

Rabit assumes the number of workers is fixed so if somehow the failed worker cannot be recovered, e.g. due to lack of
resources, then the whole Rabit process will be stuck. The network topology that Rabit constructs can only be recovered
instead of being modified based on the number of available workers. In other words, the fault tolerance of Rabit cannot
support elastic scheduling.

Rabit supports many networking options through its [MPI support](https://github.com/dmlc/rabit/blob/master/src/engine_mpi.cc)
which is not fault tolerant given that the implementation is based on MPI. If fault tolerance is enabled through Rabit's [robust implementation](https://github.com/dmlc/rabit/blob/master/src/allreduce_robust.cc),
Rabit only supports TCP networking but not others like RDMA and InfiniBand. Though it provides an interface
so developers can write implementations based on other frameworks such as NCCL and Gloo that provide additional networking
options. There's an ongoing work for Gloo implementation in Rabit [here](https://github.com/dmlc/rabit/pull/113).
