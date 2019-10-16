# Parameter Server Design

## Overview

There are commonly two implementations to support distributed training. One is allreduce, the other is parameter server. Sometimes, the model size is out of the memory of a computer.  For example, many recommending and ranking models take very high-dimensional (and sparse) inputs, thus require large embedding tables. In this scenario, parameter server would be more suitable. The parameters of a model will be sharded among several nodes, which are call PS(parameter server) nodes.

Besides, since we use a distributed PS, the network bandwidth and CPU resources are increased. Thus, the communication between workers and PS, and the optimization in PS will also be speed up.

In addition, we want to launch machine learning jobs in a Kubernetes supported cluster. The pods in k8s is scheduled with priority, and could be preempted all the time. And hardware failure problem is also nonnegligible in a big distributed system. Thus, the distributed PS have to support fault-torenlence feature.

In conclusion, a parameter server with scalability and fault-tolerance is highly needed.

In the following [PS](#ps) section, we will explain the distributed PS. In [PS Fault Tolerance](#ps-fault-tolerance) section, we will explain how to support PS fault tolerance in detail.


## PS

We will distribute model parameters into multiple PS pods, which is called parameter sharding. There is a hash function that maps a parameter to a PS pod id.

There are several kinds of parameter to be handled separately:

- Very big embedding table: Embedding table is a collection of <item id, embedding vector> pairs. The embedding table name combining an item id become a key of the hash function.
- Dense tensor: The dense tensor parameter name is the key of the hash function.

We could use a simple round-robin policy, *PS(i)=hash(key) mod N*, at first. Each PS pod will store some embedding vectors and dense tensor parameter, it only holds a subset of the whole model.

We use a customized data structure called KVStore to store model parameters in PS. There is a KVStore instance in each PS pod. The KVStore instances from the PS pods form a distributed KVStore, which could be scaled easily to support a model with a large size.

We also need an optimizer to update the model parameters stored in PS pods. To update a single parameter, the optimizer needs to get the parameter, apply the gradient to the parameter, and then write the parameter back. It involves one time read and one time write. There will be huge accesses to parameters during a training job. Since each PS pod holds a subset model, it's better to make the optimization to the subset model at the same PS pod to reduce the cost of accessing parameters.

Besides KVStore and optimizer, the PS also need to provide necessary RPC services to workers. Workers will pull the latest model parameters from PS, and push gradients to PS in each iteration of training.

Thus, we propose that PS has three basic components:

| component| functionality |hardware |
| :----  |:----  |:----  |
|KVStore  | provides distributed storage of model parameters|memory|
|Optimizer | applies gradients from workers to model parameters|CPU |
|RPC servicer |servers workers to pull parameters and push gradients | network bandwidth |

Following is the architecture diagram of PS:

![pserver](../images/pserver.png)


### KVStore

The KVStore needs to support both embedding table parameter and non-embedding parameter.

Since `tf.keras.optimizer` only accept `tf.Variable` type parameter, to avoid unnecessary memory copy, we save non-embedding parameter as a `tf.Variable` directly. We use a variable DB to store all the non-embedding parameter, the key is the variable name, the value is the variable itself.

However, the embedding table parameter could not be represented by a standard `tf.Variable`. For example, in an online learing case, new item id may come in sometimes. The shape of the embedding table is not determined. Besides, we have to initialize corresponding embedding vector value on the fly for the new item id in the PS pod.

We introduce a customized data structure `EmbeddingTable`, Following is the definition of `EmbeddingTable`:

```python
class EmbeddingTable(object):
    def __init__(self, name, dim, initializer):
        self.name = name
        self.dim = dim
        self.initializer = initializer
        self.vectors = {}

    def get(self, indices):
        res = []
        for i in indices:
            if i not in self.vectors:
                value = init_value(self.initializer, self.dim)
            else:
                value = self.vectors[i]
            res.append(value)
        return res

    def set(self, indices, value):
        pass
```

The name is the embedding layer name. It uses a dictionary `vectors` to store embedding vectors, the key is item id, the value is the embedding vector.

Since embedding vectors are lazily initialized in PS, it also has `dim` and `initializer` fields. Inside the `get` interface of `EmbeddingTable `, if the id is not in the `vectors` dictionary, the corresponding value will be initialized and return back.

There could be multiple embedding table from different embedding layer. We will create an `EmbeddingTable` instance for each embedding layer. These instances are stored at a dictionary called embedding table DB. The key is embeddig layer name, the value is the embedding table itself.

Following is the definition of KVStore:

```python
class KVStore(object):
    def __init__(self):
        self.variable_db = {}
        self.embedding_table_db = {}

    def get_parameter(self, name):
        pass

    def set_parameter(self, name, value):
        pass
```

### Optimizer

The optimizer of PS is responsible for applying gradients to parameters in KVStore. Embedding table parameter needs to be handled carefully, since its not a standard `tf.Variable`. We have already implemented an [OptimizeWrapper](https://github.com/sql-machine-learning/elasticdl/blob/develop/elasticdl/python/master/optimizer_wrapper.py) to handle this. We will move it to from master to pserver part.

The optimizer supports two kinds of parameter updating strategies: sync-SGD and async-SGD. 

- In sync-SGD, the optimizer needs to wait for a certain number of gradients, and then apply the gradients to parameters. 
- In async-SGD, the `apply_gradient` function of optimizer inside will be called inside `push_gradient` RPC service directly.

### RPC Service

PS provides necessary RPC service for workers. Following is the definition of PS RPC service:

```proto
service PServer{
    rpc push_model(Model) returns (google.protobuf.Empty);
    rpc pull_variable(PullModelRequest) returns (PullModelResponse);
    rpc pull_embedding_vector(Tensor) returns (Tensor);
    rpc push_gradient(PushGradientRequest) returns (PushGradientResponse);    
}
```

**push_model**

This is a RPC service for model initialization. There is no model definition file in the PS side. Workers will initialize the model when the first mini-batch data comes in. Then it push the model to the PS side.

Following is the definition of model proto message:

```proto
message Tensor {	
    enum DataType {	
        BOOL = 0;	
        INT16 = 1;	
        INT32 = 2;	
        INT64 = 3;	
        FP16 = 4;	
        FP32 = 5;	
        FP64 = 6;	
    }	
    string name = 1;	
    DataType data_type = 2;	
    repeated int64 dim = 3;	
    bytes content = 4;	
    repeated int64 indices = 5;	
}

message EmbeddingTableInfo{	
    string name = 1;	
    repeated int64 dim = 2;	
    string initializer = 3;	
}	
message Model {	
    int64 version = 1;	
    repeated Tensor variables = 2;
    repeated EmbeddingTableInfo embedding_table_info = 3;
}
```

Since embedding tabel parameter is initialized lazily in the PS side, we have to put some meta info defined in `EmbeddingTableInfo` in the model proto message too.

**pull_variable**

Worker will pull all non-embedding parameters before a forward pass.

**pull_embedding_vector**

Embedding layer will pull needed embedding vectors from the PS inside its `call` method.

**push_gradient**

After backward pass, worker will push the gradients to the PS.

### Checkpoint

Master will send signal to PS to make checkpoint. Each PS pod will save parameters in its current KVStore to a distributed file system.

## PS Fault Tolerance

There are two scenarios of failover to be taken into consideration:

- machines get breakdown
- some pods are killed because of priority scheduling

Since PS pods has higher priority than worker pods, worker pods of one lower priority job will be killed first to satisfy another job. If we kill all worker pods of one job, this job is actually stopped.

So we only need to focus on the first scenario. We will support PS fault tolerance by relaunching any failed PS pod and recovering its model parameters from a worker pod or replica in another PS pod.

### Fixed Domain Name for PS Pod

PS provides RPC service for workers. In order to continuously provide the RPC service for workers after a PS pod relaunch, we need to fix the domain names for PS pods. When an ElasticDL job starts, the master is responsible for starting each PS pod as a Kubernetes service. Through Kubernetes service, the domain name remains the same for every PS pod even after the relaunch.

### Model Parameter Recovery after Relaunch

The relaunched PS pod will recover model parameters to continue the training process. 

For non-embedding parameters, the PS pod can recover from workers in the same way as the [model initialization](push_model).

For embedding vectors, PS creates replicas to support fault tolerance. For each PS pod *PS(i)*, it will store *M* replicas in the following *M* PS pods from *PS(i+1 % N)* to *PS(i+M % N)*. The relaunched PS pod can recover embedding vectors from one of its replicas. 

### Embedding Replica

Assume *E(i)* is the embedding vectors in PS pod *PS(i)*, it has *M* replicas which are stored in PS pods from *P((i + 1) % N)* to *P((i + M) % N)*. Also, *PS(i)* has *M* other PS pod replicas from *E((i - M) % N)* to *E((i - 1) % N)*. 

*PS(i)* maintains *M* updated embedding vector key sets *UKS_i(j) for j from 0 to M - 1*. When *PS(i)* sparsely updates its embedding vectors *E(i)*, it also add the updated embedding vector keys into these *M* sets. 

*PS(i)* also periodically synchronize the replicas stored in it from PS pods *PS((i - M) % N)* to *PS((i - 1) % N)*. The synchronization frequency can be several seconds.

Each PS will provide a gRPC service for the replica synchronization.

```proto
message SynchronizeEmbeddingRequest {
    int32 replica_index = 1;
}

message SynchronizeEmbeddingResponse {
    map<string, Tensor> embedding_vectors = 1;
}

# GRPC service for replica synchronization
rpc synchronize_embedding(SynchronizeEmbeddingRequest) returns (SynchronizeEmbeddingResponse);

# GRPC service for PS to recover embedding vectors after relaunch
rpc get_replica(SynchronizeEmbeddingRequest) returns (SynchronizeEmbeddingResponse);
```

Each PS pod has a thread dedicated to the replica synchronization:

```python
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

```python
def synchronize_embedding(self, request, _):
    synch_embeddings = elasticdl_pb2. SynchronizeEmbeddingResponse()
    # self.UKS are the M updated embedding vector key sets in current PS
    # self.embedding_vector are the embedding vectors in current PS
    with self.lock():
        assign synch_embeddings.embedding_vectors from self.embedding_vector
        self.UKS.clear()
    return synch_embeddings
    
def get_replica(self, request, _):
    replica = elasticdl_pb2. SynchronizeEmbeddingResponse()
    assign replica.embedding_vectors from self.replicas[request.replica_index]
    return replica
```
Note that PS also needs the lock for adding updated embedding vector keys into `self.UKS` after embedding table sparse updates.
