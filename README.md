# ElasticFlow

ElasticFlow is a distributed programming framework for deep learning.  Programmers write a deep learning program by implementing a single function `forward`, just like we can write an offline data processing program using MapReduce by implementing the `map` and `reduce` functions.

ElasticFlow allows users to implement `forward` using TensorFlow graph API or PyTorch.  In either case, it invokes TensorFlow or PyTorch for [autodiff](https://arxiv.org/abs/1502.05767) -- automatically derive the backward pass from `forward` to make a complete training program.

The deep learning program fitting ElasticFlow can run locally and distributedly.  The distributed training can run on MPI/SLURM, YARN, and Kubernetes.  For Kubernetes, ElasticFlow provides a [controller](https://kubernetes.io/docs/concepts/workloads/controllers/) for [elastic scheduling](https://kubernetes.io/blog/2017/12/paddle-paddle-fluid-elastic-learning/).


## Motivation

### Ease the Deep Learning Programming

Commonly used deep learning toolkits like TensorFlow and PyTorch eases the programming of deep learning programs by providing some commonly used facilities so that programmers can define their own `main` function that calls these facilities.  In particular, programmers usually define a train loop in `main` whose each step: 

1. runs the forward pass to compute *cost*,
1. runs the backward pass, automatically derived from the forward pass, to compute *gradients*, and
1. add the weighted gradients to the model parameters.

At the end of this loop, we got the estimated model parameters.  This process is known as *training*.

The program looks something like

```python
def main():
  W = Tensor() # create model parameters
  for iter in range(1000):
     x, y = load_minibatch()
     cost = cross_entropy(softmax(fc(x, W)), y)
     dW = backward(cost, y)
     W = optimize(λ, dW, W)
```

When we are to write a distributed training program, we need to do more in the loop step: in addition to updating the local model parameters, we need to aggregate gradients from various worker processes to update a global model.  Suppose that we rely on a parameter server for gradient aggregation and global model update, the training program would look like the following:

```python
def main():
  W = Tensor() # create model parameters
  for iter in range(1000):
     x, y = load_minibatch()
     cost = cross_entropy(softmax(fc(x, W)), y)
     dW = backward(cost, y)
     W = optimize(λ, dW, W)
     push_to_parameter_server(dW)
     if iter % 100 == 0:
       W = pull_from_parameter_server()
```

Indeed, the additional cost is far more than adding two calls to `push_to_parameter_server` and `pull_from_parameter_server`; we need to program the parameter server.  Such extra development work inspires us to create a complete solution for deep learning programmers.

With ElasticFlow, the program would look like

```python
import elasticflow

def model():
  return {"W" : Tensor(), "optimizer" : SGDOptimizer(λ=0.01) }
  
def forward(x, y, params):
  return cross_entropy(softmax(fc(x, params["W"])), y)
```
  
### Fault-tolerance

ElasticFlow moves a step further to support fault tolerance, which is critical for large-scale training.  Suppose that it takes 20 minutes to finish a job by running 50 processes.  It is probable that some of them fail due to hardware failure or getting preempted by the jobs scheduling mechanism.  In such cases, ElasticFlow makes sure that the job can go on other than crashing.

ElasticFlow doesn't rely on programmers to write checkpointing code.  In the following sections, we are going to explain more.

### Elastic Scheduling

Also, ElasticFlow supports elastic scheduling to minimize not only the *training time* -- the period from when the job starts to run to its completion, but also the *pending time* -- the period from the submission of the job to its start.  The idea can be explained using the following example.   Suppose that a user submits a job requiring 32 GPUs, but the cluster has only 16 idle ones.  Usually, the job would have to wait until some other jobs complete and free some GPUs.  However, on a Kubernetes cluster with a controller provided by ElasticFlow deployed, the pending time would be zero as the controller starts the job with the 16 available GPUs, and might increase the number later.  Due to that ElasticFlow's fault-tolerance feature, the job could work well with the changing amount of processes.

## Related Work

Both TensorFlow and PyTorch have their own solution to distributed training.  However, none of them could lead to fault-tolerance, thus no way to elastic scheduling.  Also, none of them are as convenient as a deep learning framework that requires only the `forward` function definition from programmers.

### Distributed TensorFlow

TensorFlow is intrinsically a distributed system. [This official document](https://www.tensorflow.org/extend/architecture) explains the design philosophy with figures.  A summarization as follows:

1. A distributed job is a client-server architecture.  The client is a process, and a server consists of a group of processes, including a master and several workers.

1. The client program constructs a graph and sends it to the master.

1. The master receives the graph, partitions the graph, and place subgraphs on workers.

1. The execution of the graph is a collaborative result from all workers.

As a result, if any worker fails during the execution, the job fails.  And, it is technically challenging to recover the job from failure, because there is no constraints on the data-dependencies among the workers.

Some projects base themselves upon the above design, for example, KubeFlow and Estimator, thus inherit the problem.

### KubeFlow

Let us refer to this [KubeFlow example program](https://github.com/kubeflow/tf-operator/blob/v0.3.0/examples/tf_sample/tf_smoke.py) for more details.

The main function parses a JSON from the environment variable [`TF_CONFIG`](https://cloud.google.com/ml-engine/docs/tensorflow/distributed-training-details#tf-config-format), which includes the following information:

1. `tf.train.ClusterSpec` that defines the job as a cluster of processes categorized into two types: worker and parameter server (PS).  An example looks like the following:

   ```python
   cluster = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                              "worker1.example.com:2222",
                                              "worker2.example.com:2222"],
                                   "ps": ["ps0.example.com:2222",
                                          "ps1.example.com:2222"]})
   ```

1. `task.type` a string, either "worker", "ps", or "master", indicating the current process's role.
1. `task.index` an integer, together with `task.type`, indices in the above `tf.train.ClusterSpec`, to identify the current process's IP and port.

During fault recovery, failed processes are likely restarted on another computer listening on another port.  The above environment parsing mechanism is insufficient to notify processes about such changes.

Moreover, this example is tediously complex.  All that it is does is to multiply two constants on each worker.  It takes only [three lines](https://github.com/kubeflow/tf-operator/blob/fac8eff892f0e8ffa331952ab2d89e0ab18d99a3/examples/tf_sample/tf_smoke.py#L61-L63) to describe this multiplication; however, this example has about 150 lines -- most to set of the running environment to run the three steps.  These extra work are error-prone, for example:

- Mapping processes listed in `tf.train.ClusterSpec` to [virtual *devices*](https://github.com/kubeflow/tf-operator/blob/fac8eff892f0e8ffa331952ab2d89e0ab18d99a3/examples/tf_sample/tf_smoke.py#L57-L59), and [create tensors](https://github.com/kubeflow/tf-operator/blob/fac8eff892f0e8ffa331952ab2d89e0ab18d99a3/examples/tf_sample/tf_smoke.py#L60-L63) on them.

- carefully set up `tf.train.replica_device_setter` for [for a process on the server-side](https://github.com/kubeflow/tf-operator/blob/fac8eff892f0e8ffa331952ab2d89e0ab18d99a3/examples/tf_sample/tf_smoke.py#L120-L122) and [the client](https://github.com/kubeflow/tf-operator/blob/fac8eff892f0e8ffa331952ab2d89e0ab18d99a3/examples/tf_sample/tf_smoke.py#L127).

KuberFlow assumes that all above kind of work is doable by deep learning programmers, who usually have a degree in machine learning, but no much practice on distributed programming.

### TenosrFlow Estimator

TensorFlow Estimator can be used to wrap up the above complications and hide them, but cannot get rid of them. The idea is that deep learning programmers write a specific model as Python class derived from `tf.estimator.Estimator`, so that users can call the interface methods: `train`, `evaluate`, and `predict`.

The function `tf.estimator.train_and_evaluate` can start a job of trainer and parameter server processes to a model encapsulated in a derived classes distributedly, as explained in this [Google Cloud blog post](https://cloud.google.com/blog/products/gcp/easy-distributed-training-with-tensorflow-using-tfestimatortrain-and-evaluate-on-cloud-ml-engine).  However, the distributed training depends on setting up the `TF_CONFIG` like described in the above section, as represented by [the source code](https://github.com/tensorflow/tensorflow/blob/c19e29306ce1777456b2dbb3a14f511edf7883a8/tensorflow/python/estimator/training.py#L349-L374); therefore, not fault tolerable.

### TensorFlow AllReduce

[TnesorFlow AllReduce](https://github.com/baidu-research/tensorflow-allreduce) is a project contributed by Baidu Silicon Valley Research.  Instead of using parameter servers to aggregate gradients from trainer processes, we can use MPI's [`AllReduce` function](https://www.mpich.org/static/docs/v3.1/www3/MPI_Allreduce.html).  Baidu contributed their optimized version of MPICH to use sophisticated InfiniBand networks and the RDMA interface.

Programming using MPI assumes that the IP and port of each process in a job are static, just like what KubeFlow and distributed TenosrFlow require.  And, `AllReduce` fails if any process fails.

### Distributed PyTorch

To support distributed training, PyTorch provides a package [`torch.distribute`](https://pytorch.org/tutorials/intermediate/dist_tuto.html), which inherits the MPI assumptions and interface, including `send`, `recv`, and `allreduce`, and is not fault-tolerant and cannot be extended to support elastic scheduling.

## Design

### Framework

### Fault-tolerance and Elastic Scheduling
