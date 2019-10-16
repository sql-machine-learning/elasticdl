# Design Doc: Parameter Server

## Overview

In distributed machine learning, master-worker-Parameter Server (PS) architecture is commonly used. In this framework, machines are designated as three roles, one master, multiple workers and a PS with one or more PS nodes. The master is mainly responsible for assigning data to all workers. Workers are responsible for doing forward-pass and backward-pass calculation. The PS is responsible for managing model parameters. In each iteration, workers pull parameters from the PS before the forward-pass and push gradients to the PS after the backward-pass. The PS should update model parameters using gradients received.

There are two problems with the single node PS:

* For models with huge embedding tables, models may not fit in the memory of a single machine. This is especially common in the scenario of models with huge embedding table. In these models, embedding tables may be larger than 1T and model parameters excluding embedding table can fit in a single machine.
* Communication and I/O bandwidth between workers and the PS node may limit the training speed. Multiple workers may pull parameters or push gradients at the same time. However, the bandwidth of a single machine is limited. Thus, with single PS node, communication may be the bottleneck, especially when using a large number of workers or using a big model.

Therefore, ElasticDL proposes to implement a PS with multiple PS nodes and place each prarameter of the model on one of the PS nodes. In this setup, we can solve the first problem and alleviate the second problem:

* For models with huge embedding tables, every PS node only contains a subset of embedding tables. Every worker only contains all non-embedding parameters and embedding parameters that are used in one iteration, which only account for a small proportion of the embedding table. Thus, parameters on every PS node and every worker can fit in the memory of single machine.
* Everytime workers pull parameters (or push gradients), it communicates with all the PS nodes to get all model parameters (or push all gradients). Thus if the PS have *N* nodes, each worker only uses *1/N* bandwidth of each PS node. Therefore, the PS with multiple PS nodes can alleviate the communication bottleneck.

Also, using the PS with multiple PS nodes can accelerate communication because workers pull parameters concurrently from all the PS nodes, rather than pulling them serially in the single PS node setting.

Fault tolerance is an important feature of ElasticDL. The PS with multiple PS nodes can support fault tolerance by making replicas. In case a PS node fails, the master will relaunch it and recover its model parameters. The relaunch will succeed so long as there are running workers in the cluster because PS nodes have a higher priority than workers. There are two kinds of model parameters to recover, non-embedding parameters and embedding parameters. Since each worker contains all non-embedding parameters and a small propotion of embedding parameters, relaunched PS node can recover non-embedding parameters from any worker and can not recover embedding parameters from workers.

Therefore, a natural idea is to create replicas for embedding parameters and relaunched PS node can recover from replicas. In fact, Redis uses similiar strategy to support fault tolerance. More specifically, each PS node can create one or more replicas. Each replica contains the copy of all the embedding parameters saved in this PS node and each PS node save its replicas in other PS nodes. Everytime each PS node updates its embedding parameters, it should also update the replicas saved in other PS nodes. This does lead to a certain degree of decelerating training speed. This is the cost of fault tolerance. ElasticDL adopts some strategies that can minimize the efficiency influence of making replicas. More details can be found in [PS Fault Tolerance](#ps-fault-tolerance) section.

Please note that ElasticDL is a Kubernetes-native framework, and runs the master, worker nodes and PS nodes as pods in Kubernetes cluster, in where pod is the smallest deployable object. Thus in this doc, we will call them the master pod, worker pods and the PS with multiple PS pods.

## PS
As introduced above, the PS is repsonsible for managing model parameters. Thus, it is natural that each PS pod has a custom storage for model parameters and a RPC servicer to handle workers' pull parameter and push gradients requests. Also, it is better to put the optimizer used to update parameters in the same pod as the storage in order to avoid network communication (e.g. pushing updated parameters to storage).

| component| functionality |
| :----  |:----  |:----  |
|KVStore  | provides storage of model parameters|
|Optimizer | uses gradients from workers to update model parameters|
|RPC servicer |servers for the workers' pull parameters and push gradients requests|

This section will introduce the KV store and the optimizer. Since RPC servicer will involve interaction among the master, workers and the PS. It will be introduced in the [Interactions among the master, workers and PS](#Interactions-among-the-master,-workers-and-PS) section.

### KV Store
PS storage stores two kinds of parameters, embedding parameters and non-embedding parameters.

* Embedding parameters consist of mutliple embedding tables. Each embedding table corresponds to one embedding layer in model structure. An embedding table is a data structure that maps a discrete value, *id*, to a 1-d vector *vector*. This can be saved as the form of `dictionary{layer name, dictionary{id, vector}}`.
* Non-embedding parameters are in the form of mutiple TensorFlow variables in worker. The PS only needs to save the *name* and *value* of those variables. Because the *name* of variable is a unqiue identifier, non-embedding parameters are inherently suitable for KV storage.

Here is a detail that ElasticDL saves the non-embedding parameters in the form of `dictionary{name, TensorFlow.Variable}`. This is TensorFlow optimizer only accepts `tf.Variable` and transformation between `tf.Variable` and `numpy.ndarray` will copy the array object and thus waste time.

We can define the custom storage of PS as following:

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
        self.embeddings = {} # maps `id` to numpy 1-d array
    
    def get(self, ids):
        pass
        
    def set(self, ids, values):
        pass
```

### Optimizer

The optimizer of the PS is responsible for applying gradients to parameters in KV store. The optimizer should support two kinds of parameter updating strategies: 

* Synchronous SGD: The optimizer should wait for a certain number of gradients, and then use those gradients to update parameters.
* Asynchronous SGD: The optimizer should update parameters whenever it receives a gradient. More details can be found in the ElasticDL's [asynchronous SGD design doc](async_sgd_design.md).

Please note that there are some details need to handle carefully when updating embedding parameters, e.g. sparse update of embedding table, and managing slot variables in TensorFlow optimizers. ElasticDL has already implement an [OptimizeWrapper](../elasticdl/python/master/optimizer_wrapper.py) to hanldle them. We will move it from master to pserver part.


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


## Interactions among the master, workers and PS

This section will introduce the interactions among the master, workers and PS, and some related details. The following events involve interactions of these pods:

* The master starts PS.
* Initialize model parameters.
* Parameter sharding.
* Workers pull model parameters from the PS.
* Workers push gradients to the PS.

### The master starts the PS
When an ElasticDL job starts, `master.main` is responsible for starting PS as a Kubernetes service. Through Kubernetes service, we can fix domain name for every PS node.

After starting PS, `master.main` starts the master servicer and workers, and tells them the domain names of all PS nodes. For PS with embedding replicas, every PS node also needs to know the domain name of its replicas.

### Initialize model parameters
After starting, PS does not contain any parameter. For non-embedding parameters, ElasticDL should initialize them before the training process. For embedding parameters, ElasticDL adopts lazy initialization, i.e. initialize them when they are needed in the training process.

As introduced above, each worker contains all non-embedding parameters while a single PS node contains only a subset of non-embedding parameters. Thus, for models without any embedding layer, the memory of single PS pod, which is specified by the user, can be smaller than the memory of single worker pod. Therefore the model parameters may not fit in the memory of single worker pod. It is better to randomly initialize non-embedding parameters in worker pods. Workers pods push these parameters to the PS.

Here is the RPC call definition for pushing initialized model.

```proto
service PServer{
    rpc push_model(Model) returns (google.protobuf.Empty);
}
```

For embedding parameters, ElasticDL adopts lazy initialization. Thus, when an worker pull embedding vectors from PS nodes, some PS nodes may find that the worker requires some embedding vectors that does not exist in the KV store. PS nodes will initialized these embedding vectors and return them to the worker. 

Here is the pseudocode for pulling embedding parameters:

```python
service PServer{
    rpc pull_embedding_parameters(PullEmbeddingParamRequest) returns (Tensor);
}

class PServer(elasticdl_pb2_grpc.PServerServicer):
    def pull_embedding_parameters(self, req, _):
        layer_name = req.layer_name
        ids = req.ids
        return self.KVStore_instance.get_embedding_param(layer_name, ids)

class KVStore(object):
    def get_embedding_param(self, layer_name):
        self.embedding_param_db[layer_name].get(ids)

class EmbeddingTable(object):
    def __init__(self, name, dim, initializer):
        self.name = name
        self.vectors = {}
        # dim and initializer is used for random intialize embedding vectors
        self.dim = dim # the dimension of embedding vectors
        self.initializer = initializer # the initializer method of embedding vectors

    def get(self, indices):
        res = []
        for i in indices:
            if i not in self.vectors:
                value = init_value(self.initializer, self.dim)
            else:
                value = self.vectors[i]
            res.append(value)
        return res
```

### Parameter Sharding
Strictly, parameter sharding is not an interaction between pods. But it is a common functionality that is used by pull parameters request and push gradient request.

As introduced above, ElasticDL places each parameter into one of the multiple PS pods, which is called parameter sharding. Parameter sharding strategy is usually a hash function that maps a parameter to a PS pod id. For non-embedding parameter, the key of hash function is its name. For embedding parameter, the key of hash function is its embedding layer name combined with its embedding id. We could use a simple round-robin policy, *PS(i)=hash(key) % N*, at first.

### Workers pull parameters from the PS
Since ElasticDL saves parameters in PS, workers should pull parameters from the PS in each iteration of training/evaluation process.

For non-embedding parameters, we can pull all non-embedding parameters from the corresponding PS nodes before the forward-pass.

For embedding parameters, ElasticDL should only pull embedding vectors that are used in this iteration. This is because embedding vectors used in each iteration only account for a small proportion of the embedding parameters. Only when the `call` function of the embedding layer is called do we know which embedding vectors will be used in this function. Thus, the embedding layer is responsible for pulling embedding vectors from PS in its `call` function. 

Currently, ElasticDL has already implemented its embedding layer in `elasticdl.layers.embedding` module.

Here are the RPC call definitions for pulling non-embedding parameters and pulling embedding vectors.

```proto
service PServer{
    rpc pull_non_embedding_param(PullModelRequest) returns (PullModelResponse);
    rpc pull_embedding_vectors(Tensor) returns (Tensor);
}
```

### Workers push gradients to the PS
As introduced above, after backward-pass, a worker shards the gradients in the same way as the corresponding parameters and push them to PS pods. PS is responsible for using these gradients to update model parameters. 

Here is the RPC call definition for pushing gradients.

```proto
service PServer{
    rpc push_gradient(PushGradientRequest) returns (PushGradientResponse);
}
```
