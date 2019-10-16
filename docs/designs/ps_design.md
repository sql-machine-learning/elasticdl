# Design Doc: The Parameter Server

This document is about the parameter server of ElasticDL -- a Kubernetes-native and fault-tolerable distributed deep learning system.

## Background

The parameter server is one of the two commonly-used mechanisms for gradient aggregation and model updates in deep learning. The other one is AllReduce, about which we will talk in other design documents.

Comparing with AllReduce, which runs fast by utilizing highly efficient, usually ring-based, collaborative communication algorithms and advanced devices like RDMA, parameter servers can handle models with large parameter tensors by sharding them among multiple server instances.

Large parameter tensors are well-know in recommending and ranking models, which usually take high-dimensional sparse input vectors, and require large embedding tables to map the input into dense intermediate outputs. These embedding tables might be up to terabytes and run out of the memory space of any single process. We need to partition each large tensor into smaller pieces and distribute them onto multiple parameter server instances. Such distribution is known as model sharding.

Model sharding benefits performance, even if the model parameters are not very big.  Suppose all parameters of a model are on a single parameter server instance; all workers would communicate with this instance and make it a bottleneck.  With model sharding, the access and update of parameters distribute to multiple server instances.

In the literature, we see works about parameter server designs that handle large models and improve performance. However, ElasticDL needs an additional property -- fault-tolerance. The rest of the document will focus on how to achieve the three goals simultaneously:

model sharding
high performance
fault-tolerance

## Model Sharding

We consider two kinds of model parameters and their shardings:

dense tensors, and
embedding tables

Theoretically, dense tensors might have tremendous size and require sharing. However, in practices, limited by the amount of GPU memory, researchers don't often define models depending on huge dense tensor parameters. Hence in this design, we don't partition dense tensors; instead, we place each dense tensor on a parameter server instance.

Embedding tables could be huge. Since an embedding table is a map from a feature ID to an embedding vector, we might place embedding vectors in a table to different parameter server instances.

For each dense tensor or an embedding vector, denoted by x, we put it on the parameter server p(x).  For a dense tensor, x is its name; for an embedding vector, x consists of the name of the embedding table and the feature ID of the embedding vector. Please be aware that **this design assumes that each parameter has a name**.

There are many ways to define p(x). For simplicity, we chose a fixed mapping from x to a range of server instances [0, N]:

p(x) = hash(x) % N

It is noticeable that Kubernetes might preempt some parameter server instances. In such a case, we might be afraid that N isn't constant. However, we can overcome this case by setting parameter server instances having higher priority than worker processes in a job. By doing so, preemption kills workers other than parameter servers. If all workers are dead, the job stops until there come free resources, and Kubernetes starts some workers for the job.  For more information about preemption, please refer to [this document](https://kubernetes.io/docs/concepts/configuration/pod-priority-preemption/). In short, we can assume that N is a constant number in the Kubernetes-native architecture of ElasticDL.

## High Performance

As we place a tensor or an embedding vector on a parameter server instance, we need to make sure that workers can identify and access it. The placement function p(x) returns the parameter server instance, in which, we need a map data structure mapping x to the real data v. There are many high-performant mapping data structures, for example, hash map and R&B tree.

We can further improve the access performance by analyzing the querying patterns. A typical kind of querying is, given feature IDs {x₁, ..., xₜ} that appear in a sparse input vector and weights {w₁, ..., wₜ}, to return a combination of the corresponding embedding vectors, e(x₁), ..., e(xₜ), say, the weight summation Σₜ wₜ e(xₜ).  By computing the combination on parameter servers, each instance returns one vector, instead of many.

Denote the query by a sparse vector, 𝒙={xₜ : wₜ}, and the partial combinator by Θ, a worker can call a batch query API `lookup(𝒙, Θ)`, which sends 𝒙 and Θ to the set of parameter server instances p(𝒙). In each instance, denote the map data structure by M:x→e(x), it runs the following algorithm and returns a vector.

```

def lookup(𝒙, Θ):
    for x, w in 𝒙:
        if x in M:
            r = Θ(r, w e(x))
    return r
```

The worker needs to combine results from all parameter servers in p(𝒙) to get the final result r.

```
for p in p(𝒙):
    r =  Θ(r, p.lookup(𝒙, Θ))
```

Another access pattern is the optimization and update of the model parameters. By sending gradients to the parameter server instance where the corresponding tensors and embedding vectors are, and runs the optimization algorithm on parameter servers, we distribute the computation.
