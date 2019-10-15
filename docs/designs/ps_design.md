# Parameter Server Design


## Overview
In order to support the scalability of model size and PS fault tolerance in ElasticDL, ElasticDL needs a distributed parameter server. 

If a model has a very large size, it may not fit in a single parameter server (PS) memory. Processing all model parameters in a single PS requires huge computation and I/O bandwidth when the model size becomes very large. By distributing the model parameters into multiple PS nodes, a distributed PS can solve the memory, the computation and the I/O bandwidth issues when scaling up the model size.

The master can monitor PS nodes status similar to what it does for workers. In case a PS node fails, the master will relaunch it. Since PS nodes have higher priority than workers, if there are still some running workers, the relaunch will succeed by using either idle or preempted Kubernetes resources. 

After the relaunch of a PS node, the PS node needs to recover the model parameters distributed in it. There are two kinds of model parameters in ElasticDL.

* variable: trainable TensorFlow variable defined in a Keras model.
* embedding vector: embedding vector in an [embedding layer](../../elasticdl/python/elasticdl/layers/embedding.py)

Each worker has a local copy of all variables. The relaunched PS node can recover variables from any of the workers. For embedding vectors, PS must create replicas to support fault tolerance. Each PS node has its embedding vector replicas stored in other PS nodes. The relaunched PS node can recover embedding vectors from its replicas.

In the following [PS](#ps) section, we will explain the distributed PS. In [PS Fault Tolerance](#ps_fault_tolerance) section, we will explain how to support PS fault tolerance in detail.


## PS

To update a single parameter, optimizer needs to get the parameter, apply the gradient to the parameter, and then write the parameter back. It involves one time read and one time write. There will be huge accesses to parameters in a training job. Since model parameters are store at PS, it's better to make the optimization at the same place to reduce the cost of accessing parameters.

Thus, we propose that PS has three basic components:

| component| functionality |hardware |
| :----  |:----  |:----  |
|KVStore  | provides distributed storage of model parameters|memory|
|Optimizer | applies gradients from workers to model parameters|CPU |
|RPC servicer |servers workers to pull parameters and push gradients |network bandwidth |

Each component will occupy a kind of hardware resource. The hardware resources of PS nodes, including memory/CPU/network bandwidth, are fully used.

Following is the architecture diagram of PS:

![pserver](../images/pserver.png)

There is a KVStore instance in each PS node. The KVStore instances of all the PS nodes combine together and become a distributed KVStore, which could be expanded easily to support the large size model.

Correspondingly, there are an optimizer instance and a RPC servicer instance in each PS node. The optimizer gets gradients for RPC servicer, parameters from KVStore, and then applies the gradients to parameters. At last, it updates parameters back to the KVStore.

### KVStore

The typical big parameter of a model could be an embedding table. There could be billions of feature ID in a recommendation system. The KVStore needs to support both embedding table parameter and non-embedding parameter.

Since `tf.keras.optimizer` only accept `tf.Variable` as the parameter. To avoid uncessary memory copy, we save non-embedding parameter as a `tf.Variable` in variable DB directly.

For the embedding table parameter, we introduce a customized data structure because we could not initialize the table before training. There are mainly two reasons:

- Feature ID is usually calculated by the hash function. The training dataset is too large to get the range of the ID before training.
- Sometimes, there comes a new feature ID in online training.


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

    def get_embedding_vector(self, name, indices):
        pass

    def set_embedding_vector(self, name, indices, value):
        pass
```

Following is the definition of EmbeddingTable:

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

The name of embedding table is the embedding layer name. EmbeddingTable uses a dictionary `vectors` to store `<id, embedding_vector>` pairs.

Since embedding table is lazily initialized in PS, it also has `dim` and `initializer` fields. Inside the `get` interface of `EmbeddindTable`, if the id is not in the `vectors` dictionary, the corresponding value will be initialized and return back.

Then, all the embedding tables are stored at the embedding table DB. The key is the usually the embedidng layer name, the value is the embedidng table parameter.

### Optimizer

#### embedding parameter and non-embedding parameter updating

The optimizer of PS is responsible for applying gradients to parameters in KVStore. Non-embedding parameter in KVStore is stored as a `tf.Variable`, we could set a `tf.keras.optimizer` instance in the optimizer to update it.

However, embedding table parameter is not a standard `tf.Variable`, we have to implement extra logic to handle it. The optimizer need to `get_embedding_vector` from the KVStore, and convert it to a standard `tf.Variable` on the fly.

Besides, many `tf.keras.optimizer` subclasses, such as `Adam` and `Adagrad` allocate and manage additional variables associated with the variables to train.  These are called `Slots`. Non-embedding parameter slots are stored and managed by `tf.keras.optimizer`.

Embedding table slots are stored at KVStore. It's also a embedding table data structure. For example, a embedding table parameter with name `embedding_layer0`, we will create a corresponding `embedding_layer0-momentum` EmbeddingTable object in `KVStore.embedding_table_db`.

We also need to implement extra logic to handle emebdding table slots when doing optimization.

#### sync-SGD and async-SGD

We support sync-SGD and async-SGD both.

In sync-SGD, optimizer needs to wait for a certain number of gradients, and then get the gradient after addition. We could implement a customized gradient queue structure to support such logic efficiently.

There are two ways to support async-SGD:
 
- Calling `apply_gradient` inside `push_gradient` gRPC service.
- Putting gradients into a gradient queue, and optimizer gets gradients from the queue immediately to `apply_gradient`.

In the first way, there may be several gRPC threads running in parallel. This will introduce race condition on parameter updating, some gradients may be overwrited.

The second way ensure each gradient could be applied, and decoupling these two procedures, `push_gradient` of worker and `apply_gradient` of optimizer. But the second way introduces more staleness in updating model, and may influence the final training accuracy.

We will choose the first way since it's more easier to implement.

### RPC Service
PS provides RPC service for workers. There are three important events that PS should interact with workers:

* initialization of model parameters
* pull model parameters
* push gradients

#### Initialization of model parameters
After starting, PS does not contain any parameter. For model variables, ElasticDL should initialize them before training process. For embedding vectors, ElasticDL adopts lazy initialization, i.e. initialize them when they are needed in training process.

In distributed learning scenario, model variables can be very large and each PS node only contains a subset of model variables. We can't assume that memory of single PS node can store all model variables. Thus, workers are responsible for model variables random initialization. After random initialization, workers sends model variables to the corresponding PS node.

Here is the RPC call definition for pushing initialized model.

```proto
service PServer{
    rpc push_model(Model) returns (google.protobuf.Empty);
}
```

#### Pull model parameters
Since ElasticDL saves parameters in PS, workers should pull parameters from PS in each iteration of training/evaluation process.

For model variables, we can simply pull all model variables in one gRPC call before the forward-pass.

For embedding vectors, we assume that the embedding vectors used in each iteration only account for a very small proportion of the embedding tables. Due to this assumption, ElasticDL should only pull embedding vectors that are used in this iteration. Only when the `call` function of embedding layer is called do we know which embedding vectors will be uses in this function. Thus, embedding layer is responsible for pull embedding vectors from PS. 

Currently, ElasticDL has already implemented its embedding layer in `elasticdl.layers.embedding` module.

Here are the RPC call definitions for pulling model variables and pulling embedding vectors.

```proto
service PServer{
    rpc pull_variable(PullModelRequest) returns (PullModelResponse);
    rpc pull_embedding_vector(Tensor) returns (Tensor);
}
```

#### Push Gradients
As introduced above, after backward-pass, workers push gradients to PS. PS is responsible for using these gradients to update model parameters.

Here is the RPC call definition for pushing gradients.

```proto
service PServer{
    rpc push_gradient(PushGradientRequest) returns (PushGradientResponse);
}
```

### Checkpoint and Serving

Master will send signal to PS to make checkpoint. Each PS node will save parameters in its current KVStore to a distributed file system.

Since a PS node only has a subset of the whole model, we have to merge these submodels to get final model for serving.

## Interactions among Master, PS and Worker
The following events involve interactions among the master and PS:

* The master starts PS.
* PS reports submodel version to the master.
* The master tells PS to save checkpoint.

### The master starts PS
When an ElasticDL task starts, `master.main` is responsible for starting PS as a Kubernetes service. Through Kubernetes service, we can fix domain name for every PS node.

After starting PS, `master.main` starts the master servicer and workers, and tells them the domain names of all PS nodes. For PS with embedding replicas, every PS node also needs to know the domain name of its replicas.

### PS reports submodel version to the master

The master needs to know the model version to decide when to save checkpoint and when to evaluate model. PS regularly reports the version of the submodel it possessed to the master.


## PS Fault Tolerance


### Relaunch of PS Node

The master will create a distributed PS with *N* PS nodes, where *N* is specified by the user. In case a PS node fails, the master will relaunch it by Kubernetes APIs. As we discussed in [Overview](#overview), the relaunch will succeed as long as there are still running workers.

The relaunched PS node will recover model parameters to continue the training. 
For model variables, the PS node can recover from workers in the same way as the variable initialization by setting its variable status as uninitialized.

For embedding vectors, PS creates replicas to support fault tolerance. For each PS node *PS(i)*, it will store *M* replicas in the following *M* PS nodes from *PS(i+1 % N)* to *PS(i+M % N)*. The relaunched PS node can recover embedding vectors from one of its replicas. 


### Embedding Replica

Assume *E(i)* is the embedding vectors in PS node *PS(i)*, it has *M* replicas which are stored in PS nodes from *P((i + 1) % N)* to *P((i + M) % N)*. Also, *PS(i)* has *M* other PS node replicas from *E((i - M) % N)* to *E((i - 1) % N)*. 

*PS(i)* maintains *M* updated embedding vector key sets *UKS_i(j) for j from 0 to M - 1*. When *PS(i)* sparsely updates its embedding vectors *E(i)*, it also add the updated embedding vector keys into these *M* sets. 


*PS(i)* also periodically synchronize the replicas stored in it from PS nodes *PS((i - M) % N)* to *PS((i - 1) % N)*. The synchronization frequency can be several seconds.

Each PS will provide a gRPC service for the replica synchronization.

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

Each PS node has a thread dedicated to the replica synchronization:

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
