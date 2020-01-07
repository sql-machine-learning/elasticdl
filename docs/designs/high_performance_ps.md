# High Performance Parameter Server Design


## Motivation

This design doc focus on implementing a high performance parameter server(short for PS). For the functionality of the PS, please refer to this [design doc](https://github.com/sql-machine-learning/elasticdl/blob/develop/docs/designs/parameter_server.md)

PS receives gradients from workers, applies gradients to parameters, and sends the latest parameters to workers. Receiving gradients and sending parameters bring IO workload to PS, and applying gradients to parameters brings CPU workload to PS. Since one PS could receive gradients from mant workers, both IO workload and CPU workload would be very high.

The current PS is implemented with Python. Because of `GIL` of Python, gradients are applied to parameters sequentially with only one CPU core. As a result, the receiving gradients service is also blocked, and waiting for current gradients to be consumed. To resolve this bottleneck, we have to fully use multi CPU cores capability of PS.

Usually, the first thing that comes to mind is using C++ to re-implement such a high performance parameter server. But we have some concerns on the development efficiency of C++. Golang is another potential choice. In this doc, we will go through the key points of implementing a high performance parameter server to see if Golang is competent for the job and could substitute C++.


## Communication

The PS provides services to workers with gRPC library. Both C++ and Go are well supported in gRPC. The development efficiency of C++ could be some less than Go.

## Computation

The gradients and parameters on PS are represented by tensors. And applying gradients to parameters, which is also called optimization, is acutally an operation of tensors.

### Tensor

We have to support both dense tensor and sparse tensor. Besides, different element data types are also needed, such as int8/int32/float16/float32/float64. Int8 and float16 is used in training based quantization. The tensor operators have to support different data types.

C++ supports generic with template programming, while Go does not support generic directly.

### Math library

There are different kinds of optimizers, which need some tensor operations. There are many mature math libraries developed with C++. For example, eigen is used in TensorFlow and Paddle, aten is used in Pytorch. These math libraries provide abundant tensor operators and support both CPU and GPU. Besides, these math libraries could call some state-of-the-art blas libraries internally, such as MKL and cuBLAS. With these math libraries, the optimization operators could be implemented easily. 

*Go part TBD*

Need to survey in further. Generally, the math library ecology of Go is far from competing to C++.


## Scheduling

In C++, we use thread based scheduling. Threads are scheduled by OS. Usually, we will implement a thread pool for computation, and another thread pool for IO. The parameter optimzation will be processed by the computation thread pool in parallel. In further, to reduce the overhead of context switching, we could binds a thread to a certain CPU core by setting CPU affinity to a thread. It will increase the cache hit rate of CPU cores.

In Go, there is no concept of thread, we use goroutine instead. Goroutines are scheduled by Go runtime. Goroutine is not preemptive. When an event like IO/function call/channel/runtime.Goshed() happens, the goroutine could be switched. There is a possibility that IO goroutines could not be scheduled for a while if all the CPU cores are occpuied by computation goroutines. We will do some experiments to check this.




