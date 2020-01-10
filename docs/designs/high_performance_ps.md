# High Performance Parameter Server Design

## Motivation

This design doc focuses on implementing a high performance parameter server(short for PS). For the functionality of the PS, please refer to this [design doc](https://github.com/sql-machine-learning/elasticdl/blob/develop/docs/designs/parameter_server.md)

PS receives gradients from workers, applies gradients to parameters, and sends the latest parameters to workers. Receiving gradients and sending parameters bring IO workload to PS, and applying gradients to parameters brings CPU workload to PS. Since one PS could receive gradients from many workers, both IO workload and CPU workload would be very heavy.

The current PS is implemented with Python. Because of [GIL](https://wiki.python.org/moin/GlobalInterpreterLock) of Python, gradients are applied to parameters sequentially with only one CPU core. As a result, the receiving gradients service is also blocked, and waiting for current gradients to be consumed. To resolve this bottleneck, we have to fully use multi CPU cores of PS.

Usually, the first thing that comes to mind is using C++ to reimplement a high performance parameter server. But we have some concerns on the development efficiency of C++. Go is another potential choice. In this doc, we will go through the key points of implementing a high performance parameter server to see if Go is competent for the job and could substitute C++ in all or in part.

## Communication

The PS provides services to workers with gRPC library. Both C++ and Go are well supported in gRPC. Go has better development efficiency than C++.

## Computation

The gradients and parameters on PS are represented by tensors. And applying gradients to parameters, which is also called optimization, is acutally a math operation of tensors.

### Tensor

We have to support both dense tensor and sparse tensor. Besides, different element data types are also needed, such as int8/int32/float16/float32/float64. Int8 and float16 are used in training based quantization.

Each tensor operator has to support different data types. C++ supports generics with template programming, while Go does not support generics directly.

### Math library

There are different kinds of optimizers, which need some tensor operations. There are many mature math libraries developed with C++. For example, [eigen](https://gitlab.com/libeigen/eigen) is used in TensorFlow and Paddle, [aten](https://github.com/pytorch/pytorch/tree/master/aten) is used in Pytorch. These math libraries provide abundant tensor operators and support both CPU and GPU. Besides, these math libraries could call some state-of-the-art blas libraries internally, such as MKL and cuBLAS. With these math libraries, the operators in optimizers could be implemented easily and efficiently.

It seems that there are few math libraries in Go. [gosl](https://github.com/cpmech/gosl) is no longer active, and [gonum](https://github.com/gonum/gonum) does not support MKL. Generally, the math library ecology of Go is far from competing to C++. And we also has some faint worry with the performance of math libraries in Go.

## Scheduling

In C++, we use thread based scheduling. Threads are scheduled by the operating system. Usually, we will implement a thread pool for computation, and another thread pool for IO. The parameter optimzation will be processed by the computation thread pool in parallel. In further, to reduce the overhead of context switching, we could bind a thread to a certain CPU core by setting CPU affinity to the thread. It will increase the cache hit rate of a CPU core.

In Go, there is no concept of thread, we use goroutine instead. Goroutines are scheduled by Go runtime. Goroutine is not preemptive. There are four classes of events that occur in Go programs that allow the scheduler to make scheduling decisions. This does not mean it will always happen on one of these events. It means the scheduler gets the opportunity.

- The use of the keyword `go`
- Garbage collection
- System calls
- Synchronization and Orchestration

Go supports concurrent programming well with first-class concepts, goroutine and channel.

## Conclusion

Considering the tradeoff between development efficiency and program peformance, we plan to put communication and scheduling parts in Go, and computation in C++.

[Cgo](https://golang.org/cmd/cgo/) enables the creation of Go packages that call C code. And the overhead of cgo is slight. The optimization operators will be implemented in C++, wrappered with C interface, and exposed to Go.

The receiving gradients and sending parameters service are implemented in Go. Once receving gradients from a worker, a gouroutine will be launched to do optimization.

## Reference

- https://gitlab.com/libeigen/eigen
- https://github.com/cpmech/gosl
- https://github.com/gonum/gonum
- https://www.ardanlabs.com/blog/2018/08/scheduling-in-go-part2.html