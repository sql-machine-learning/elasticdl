# ElasticFlow

ElasticFlow is a distributed programming framework for deep learning.  Programmers write a deep learning program by implementing a single function `forward`, just like we can write an offline data processing program using MapReduce by implementing the `map` and `reduce` functions.

ElasticFlow allows users to implement `forward` using TensorFlow graph API or PyTorch.  In either case, it invokes TensorFlow or PyTorch for [autodiff](https://arxiv.org/abs/1502.05767) -- automatically derive the backward pass from `forward` to make a complete training program.

The deep learning program fitting ElasticFlow can run locally and distributedly.  The distributed training can run on MPI/SLURM, YARN, and Kubernetes.  For Kubernetes, ElasticFlow provides a [controller](https://kubernetes.io/docs/concepts/workloads/controllers/) for [elastic scheduling](https://kubernetes.io/blog/2017/12/paddle-paddle-fluid-elastic-learning/).


## Motivation

### Ease the Deep Learning Programming

Commonly used deep learning toolkits like TensorFlow and PyTorch eases the programming of deep learning programs by providing some commonly used facilities so that programmers can define their own `main` function that calls these facilities.  In particular, programmers usually define a train loop in `main` whose each step 

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

## Design

### Framework

### Fault-tolerance and Elastic Scheduling
