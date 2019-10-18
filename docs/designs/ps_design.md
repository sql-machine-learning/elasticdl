# Design Doc: Parameter Server
This document is about the parameter server of ElasticDL. ElasticDL is a Kubernetes-native and fault-tolerable distributed deep learning system.

## Background
The parameter server (PS) is one of the two commonly-used mechanisms for gradient aggregation and model updates in deep learning. The other one is AllReduce, about which we will talk in other design documents.

In PS mechanism, machines are designated as three roles, one master, multiple workers and a PS with one or more PS instances. In the training process, workers pull parameters from the PS and push gradients to the PS. The PS should update model parameters using gradients received.

It is noticeable that most designs use multiple PS instances to form a *service* in each job. These instances can collaboratively maintain large models with model sharding. Sometimes, even when the model is not very big, users still want multiple PS instances to benefit performance, because a single PS instance might become the bottleneck of communication and computation.

In literatures, we see works about parameter server designs that handle large models and improve performance. However, ElasticDL needs an additional property -- fault-tolerance. The rest of the document will focus on how to achieve the three goals simultaneously:

1. model sharding
2. high performance
3. fault-tolerance

## Model Sharding
There are two kinds of parameters in the training process:

* Embedding parameters consist of mutliple embedding tables. Each embedding table corresponds to one embedding layer in model structure. An embedding table is a data structure that maps a discrete value, named embedding id *id*, to a 1-d vector, named embedding vector *vector*. 
* Non-embedding parameters are in the form of multiple dense tensors. In order to distinguish these dense tensors, TensorFlow assigns a unique name to each tensor (or in the form of TensorFlow variables).

For embedding parameters, it is a natural idea to distribute different embedding vectors on different parameter server instance.

For dense tensors, theoretically, they might have tremendous size and require sharding. However, in practices, researchers don't often define models depending on huge dense tensor parameters. Hence in this design, we don't partition dense tensors; instead, we place each dense tensor on a parameter server instance.

For each dense tensor or an embedding vector, denoted by x, we put it on the parameter server p(x). For a dense tensor x, we denote key(x) for its name; for an embedding vector x, key(x) consists of the name of the embedding layer and its embedding id.

There are many ways to define p(x). For simplicity, we choose a fixed mapping from x to a range of PS instances [0, N):

p(x) = hash(key(x)) % N

It is noticeable that Kubernetes might preempt some parameter server instances. In such a case, we might be afraid that N isn't constant. However, we can overcome this case by setting parameter server instances having higher priority than worker processes in a job. By doing so, preemption kills workers other than parameter servers. If all workers are dead, the job stops until there comes free resources, and Kubernetes starts some workers for the job.  For more information about preemption, please refer to [this document](https://kubernetes.io/docs/concepts/configuration/pod-priority-preemption/). In short, we can assume that N is a constant number in the Kubernetes-native architecture of ElasticDL.

## High Performance
As introduced above, when using a large model and a large number of workers, the key to high performance of PS mechanism is achieving efficient communication between workers and the PS. In order to achieve this goal, we need to have a reasonable parameter storage scheme first, then we can design an efficient scheme to pull parameters, push gradient and update parameters.

### Parameter Storage
Parameter storage includes two kinds of parameters:

* Embedding parameters can be saved in the form of `dictionary{layer name, dictionary{id, vector}}`.
* Each Non-embedding parameter consists of a name and a dense tensor, which is inherently suitable for KV storage.

There are many high-performant technique for KV storage, for example, hash map and R&B tree. For simplicity, we can use python dictionary at first. Hence we can save parameters in the following structure:

```python
class KVStore(object):
    def __init__(self):
        self.non_embedding_param_db = {} # maps `variable_name` to TensorFlow variable instance
        self.embedding_param_db = {} # maps `layer_name` to `EmbeddingTable` instance 

    def get_non_embedding_param(self, names):
        pass
    
    def get_embedding_param(self, layer_name):
        pass

    def set_non_embedding_param(self, names, values):
        pass
        
    def set_embedding_param(self, layer_name, value):
        pass

class EmbeddingTable(object):
    def __init__(self, layer_name):
        self.layer_name = layer_name
        self.embeddings = {} # maps `id` to 1-D numpy.ndarray
    
    def get(self, ids):
        pass
        
    def set(self, ids, values):
        pass
```

### Pull Parameters
Since ElasticDL saves parameters in PS, workers should pull parameters from the PS in training/evaluation process. 

For non-embedding parameters, we can pull all non-embedding parameters from the corresponding PS instances before the forward-pass.

For embedding parameters, ElasticDL should only pull embedding vectors that are used in this iteration. This is because embedding vectors used in each iteration only account for a small proportion of the embedding parameters. Only when it it time for embedding layer to do forward-pass, can we know which embedding vectors will be used in this iteration. Thus, the embedding layer is responsible for pulling embedding vectors from PS in its forward-pass process. 

Thanks to the model sharding technique, ElasticDL can implement above processes in parallel communication with all PS instances, thus achieve the goal of high performance.

Here are the RPC call definitions for pulling non-embedding parameters and pulling embedding vectors.

```proto
service PServer{
    rpc pull_non_embedding_param(PullModelRequest) returns (PullModelResponse);
    rpc pull_embedding_vectors(Tensor) returns (Tensor);
}
```

### Push Gradients
After backward-pass, a worker shards the gradients in the same way as the corresponding parameters and push them to PS instances concurrently. 

Here is the RPC call definition for pushing gradients.

```proto
service PServer{
    rpc push_gradient(PushGradientRequest) returns (PushGradientResponse);
}
```

### Update Parameters
The optimizer of the PS is responsible for using received gradients to update parameters. The optimizing process should support two kinds of parameter updating strategies, synchronous SGD and asynchronous SGD, about which we have introduced in other design documents.

To avoid communication between optimizer and the PS, we propose to put the optimizer into the parameter server pods, i.e. every parameter server instance has an optimizer instance. This can also distribute the optimization computation, thus benefit performance.

## PS Fault Tolerance

We support PS fault tolerance by relaunching any failed PS pod and recovering its model parameters. 

The master will create a PS with *N* PS pods, where *N* is specified by the user. In case a PS pod fails, the master will relaunch it by Kubernetes APIs. As we discussed in [Overview](#overview), the relaunch will succeed as long as there are still running worker pods.

### Fixed Domain Name for PS Pod

PS provides RPC service for workers. In order to continuously provide the RPC service for workers after a PS pod relaunch, we use fixed domain names for PS pods. When an ElasticDL job starts, the master is responsible for starting each PS pod as a Kubernetes service. Through Kubernetes service, we can fix domain name for every PS pod even after the relaunch.

### Model Parameter Recovery after Relaunch

The relaunched PS pod will recover model parameters to continue the training process. 

For non-embedding parameters, the PS pod can recover from workers in the same way.

For embedding parameters, every PS creates replicas to support fault tolerance. For each PS pod *PS(i)*, it will store *M* replicas in the following *M* PS pods from *PS((i+1) % N)* to *PS((i+M) % N)*. Each replica contains the copy of embedding parameters saved in *PS(i)*. Each time *PS(i)* updates its embedding parameters, it should also update the *M* replicas. The relaunched PS pod can recover embedding vectors from one of its replicas. 

### Embedding Replica

Assume *E(i)* is the embedding vectors in PS pod *PS(i)*, it has *M* replicas which are stored in PS pods from *P((i + 1) % N)* to *P((i + M) % N)*. Also, *PS(i)* has *M* other PS pod replicas from *E((i - M) % N)* to *E((i - 1) % N)*. 

Foe the strategy to update replicas, ElasticDL adopts a delayed updating strategy, which can reduce the network traffic by an order of magnitude. More specifically, *PS(i)* maintains *M* updated embedding vector key sets *UKS_i(j) for j from 0 to M - 1*. When *PS(i)* sparsely updates its embedding vectors *E(i)*, it also add the updated embedding vector keys into these *M* sets. The replicas of *PS(i)* (i.e. *PS((i - M) % N)* to *PS((i - 1) % N)*) periodically synchronize the replicas from *PS(i)*. The synchronization frequency can be several seconds.

Each PS pod will provide a gRPC service for the replica synchronization.

```
message SynchronizeEmbeddingRequest {
    int32 replica_index = 1;
}

message SynchronizeEmbeddingResponse {
    map<string, Tensor> embedding_vectors = 1;
}

# GRPC service for replica synchronization
rpc SynchronizeEmbedding(SynchronizeEmbeddingRequest) returns (SynchronizeEmbeddingResponse);

# GRPC service for PS to recover embedding vectors after relaunch
rpc GetReplica(SynchronizeEmbeddingRequest) returns (SynchronizeEmbeddingResponse);
```

Each PS pod has a thread dedicated to the replica synchronization:

```
# T is the number of seconds for synchronization frequency
# Assume current PS is PS(i), self._stub[index] is the stub for PS((i - index) % N)'s GRPC server.
# self.replicas[index] is the replica for PS((i - index) % N).
req = elasticdl_pb2.SynchronizeEmbeddingRequest()
while still training:
    time.sleep(T)
    for index in range(M):
        req.replica_index = index
        updated_vectors = self._stub[replica_index].SynchronizeEmbedding(req)
        update self.replicas[index] from updated_vectors.embedding_vectors
```

The implementation of the gRPC services:

```
def SynchronizeEmbedding(self, request, _):
    synch_embeddings = elasticdl_pb2. SynchronizeEmbeddingResponse()
    # self.UKS are the M updated embedding vector key sets in current PS
    # self.embedding_vector are the embedding vectors in current PS
    with self.lock():
        assign synch_embeddings.embedding_vectors from self.embedding_vector
        self.UKS.clear()
    return synch_embeddings
    
def GetReplica(self, request, _):
    replica = elasticdl_pb2. SynchronizeEmbeddingResponse()
    assign replica.embedding_vectors from self.replicas[request.replica_index]
    return replica
```
Note that PS also needs the lock for adding updated embedding vector keys into `self.UKS` after embedding table sparse updates.