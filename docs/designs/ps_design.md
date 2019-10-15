# Parameter Server Design


## Overview
In order to support the scalability of model size and PS fault tolerance in ElasticDL, ElasticDL needs a distributed parameter server. 

If a model has a very large size, it may not fit in a single parameter server (PS) memory. Processing all model parameters in a single PS requires huge computation and I/O bandwidth when the model size becomes very large. By distributing the model parameters into multiple PS pods, a distributed PS can solve the memory, the computation and the I/O bandwidth issues when scaling up the model size.

The master can monitor PS pods status similar to what it does for worker pods. In case a PS pod fails, the master will relaunch it. Since PS pods have higher priority than worker pods, if there are still some running worker pods, the relaunch will succeed by using either idle or preempted Kubernetes resources. 

After the relaunch of a PS pod, the PS pod needs to recover the model parameters distributed in it. There are two kinds of model parameters in ElasticDL.

* variable: trainable TensorFlow variable defined in a Keras model.
* embedding vector: embedding vector in an [embedding layer](../../elasticdl/python/elasticdl/layers/embedding.py)

Each worker has a local copy of all variables. The relaunched PS pod can recover variables from any of the workers. For embedding vectors, PS must create replicas to support fault tolerance. Each PS pod has its embedding vector replicas stored in other PS pods. The relaunched PS pod can recover embedding vectors from its replicas.

In the following [PS](#ps) section, we will explain the distributed PS. In [PS Fault Tolerance](#ps_fault_tolerance) section, we will explain how to support PS fault tolerance in detail.


## PS

We will distribute model parameter into multiple PS pods, which is called parameter sharding. The key of a variable is its name. The key of a embedding vector is its embedding layer name combining an item id. There is a hash function which maps the key to a PS pod id. Each PS pod only holds a subset of the whole model.

There is a KVStore instance in each PS pod. The KVStore instances of all the PS pods combine together and become a distributed KVStore, which could be expanded easily to support a model with large size.

We also need an optimizer to update the model parameters stored in PS pods. To update a single parameter, optimizer needs to get the parameter, apply the gradient to the parameter, and then write the parameter back. It involves one time read and one time write.There will be huge accesses to parameters during a training job. Since model parameters are store at PS, it's better to make the optimization at the same place to reduce the cost of accessing parameters.

So, workers need to pull the latest model parameters from PS, and push gradients to PS in each iteration of training.

Thus, we propose that PS has three basic components:

| component| functionality |hardware |
| :----  |:----  |:----  |
|KVStore  | provides distributed storage of model parameters|memory|
|Optimizer | applies gradients from workers to model parameters|CPU |
|RPC servicer |servers workers to pull parameters and push gradients | network bandwidth |

Each component will occupy a kind of hardware resource. The hardware resources of PS pods, including memory/CPU/network bandwidth, are fully used.

Following is the architecture diagram of PS:

![pserver](../images/pserver.png)


Correspondingly, there are an optimizer instance and a RPC servicer instance in each PS node. The optimizer gets gradients for RPC servicer, parameters from KVStore, and then applies the gradients to parameters. At last, it updates parameters back to the KVStore.

### KVStore

The typical big parameter of a model could be an embedding table. There could be billions of feature ID in a recommendation system. The KVStore needs to support both embedding vector parameter and non-embedding parameter.

Since `tf.keras.optimizer` only accept `tf.Variable` as the parameter. To avoid uncessary memory copy, we save non-embedding parameter as a `tf.Variable` in variable DB directly.

For the embedding vector parameter, we introduce a customized data structure because we could not get the determine dimension infomation, and initialize the embedding vectors before training. There are mainly two reasons:

- Feature ID is usually calculated by the hash function. The training dataset is too large to get the range of the ID before training.
- Sometimes, there comes a new feature ID in online learning.

Following is the definition of EmbeddingVectors:

```python
class EmbeddingVectors(object):
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

The name is the embedding layer name of embedding vectors. It uses a dictionary `vectors` to store `<id, embedding_vector>` pairs.

Since embedding vectors are lazily initialized in PS, it also has `dim` and `initializer` fields. Inside the `get` interface of `EmbeddingVectors `, if the id is not in the `vectors` dictionary, the corresponding value will be initialized and return back.

There could be multiple embedding vectors from different embedding layer. We save all these embedding vectors in a independent embedding vector DB of KVStore.

Following is the definition of KVStore:

```python
class KVStore(object):
    def __init__(self):
        self.variable_db = {}
        self.embedding_vector_db = {}

    def get_parameter(self, name):
        pass

    def set_parameter(self, name, value):
        pass
```


### Optimizer

The optimizer of PS is responsible for applying gradients to parameters in KVStore. Embedding table parameter needs to be handled carefully, since its not a standard `tf.Variable`. We have implemented a [OptimizeWrapper](https://github.com/sql-machine-learning/elasticdl/blob/develop/elasticdl/python/master/optimizer_wrapper.py) already to handle this. We will move it to from master to pserver part.

The optimzer supports two kinds of parameter updating strategies: sync-SGD and async-SGD. In sync-SGD, optimizer needs to wait for a certain number of gradients, and then apply the gradients to parameters. In async-SGD, the  `apply_gradient` function of optimizer inside will be called inside `push_gradient` RPC service directly.

### RPC Service
PS provides RPC service for workers. There are three important events that PS should interact with workers:

* initialization of model parameters
* pull model parameters
* push gradients

**Initialization of model parameters**

After starting, PS does not contain any parameter. For model variables, ElasticDL should initialize them before training process. For embedding vectors, ElasticDL adopts lazy initialization, i.e. initialize them when they are needed in training process.

Since a single PS pod may not have enough memory for big modle, workers are responsible for random initializing model variables. After initializing, workers push initialized model variables to corresponding PS pod. Here is the RPC call definition for pushing initialized model.

```proto
service PServer{
    rpc push_model(Model) returns (google.protobuf.Empty);
}
```

**Pull model parameters**

Since ElasticDL saves parameters in PS, workers should pull parameters from PS in each iteration of training/evaluation process.

For model variables, we can simply pull all model variables in one gRPC call before the forward-pass.

For embedding vectors, ElasticDL should only pull embedding vectors that are used in this iteration. This is because embedding vectors used in each iteration only account for a small proportion of the embedding tables. Only when the `call` function of embedding layer is called do we know which embedding vectors will be uses in this function. Thus, embedding layer is responsible for pull embedding vectors from PS in its `call` function.

Currently, ElasticDL has already implemented its embedding layer in `elasticdl.layers.embedding` module.

Here are the RPC call definitions for pulling model variables and pulling embedding vectors.

```proto
service PServer{
    rpc pull_variable(PullModelRequest) returns (PullModelResponse);
    rpc pull_embedding_vector(Tensor) returns (Tensor);
}
```

**Push Gradients**

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

## PS Fault Tolerance

We support PS falut tolerance by relaunching any failed PS pod and recovering its model parameters. 

The master will create a distributed PS with *N* PS pods, where *N* is specified by the user. In case a PS pod fails, the master will relaunch it by Kubernetes APIs. As we discussed in [Overview](#overview), the relaunch will succeed as long as there are still running worker pods.

### Fixed Domain Name for PS Pod
PS provides RPC service for workers. In order to continuously provide the RPC service for workers after a PS pod relaunch, we use fixed domain name for PS pods. When an ElasticDL task starts, the master is responsible for starting each PS pod as a Kubernetes service. Through Kubernetes service, we can fix domain name for every PS node even after the relaunch.

### Model Parameter Recovery after Relaunch

The relaunched PS pod will recover model parameters to continue the training. 
For model variables, the PS pod can recover from workers in the same way as the variable initialization by setting its variable status as uninitialized.

For embedding vectors, PS creates replicas to support fault tolerance. For each PS pod *PS(i)*, it will store *M* replicas in the following *M* PS pods from *PS(i+1 % N)* to *PS(i+M % N)*. The relaunched PS pod can recover embedding vectors from one of its replicas. 


### Embedding Replica

Assume *E(i)* is the embedding vectors in PS pod *PS(i)*, it has *M* replicas which are stored in PS pods from *P((i + 1) % N)* to *P((i + M) % N)*. Also, *PS(i)* has *M* other PS pod replicas from *E((i - M) % N)* to *E((i - 1) % N)*. 

*PS(i)* maintains *M* updated embedding vector key sets *UKS_i(j) for j from 0 to M - 1*. When *PS(i)* sparsely updates its embedding vectors *E(i)*, it also add the updated embedding vector keys into these *M* sets. 


*PS(i)* also periodically synchronize the replicas stored in it from PS pods *PS((i - M) % N)* to *PS((i - 1) % N)*. The synchronization frequency can be several seconds.

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
