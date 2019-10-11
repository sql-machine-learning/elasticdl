# Parameter Server Design


## Overview
Currently, there is one parameter server (PS) co-existed with the master. In order to support PS with multiple nodes and PS fault tolerance, we need to separate PS from the master. Besides a KV store for model parameters, PS should also support updating parameters using gradients. Currently, models have two kinds of parameter, variable and embedding vector. PS should support model variable update and embedding table sparse update.

The master will create a PS with *N* PS nodes, where *N* is specified by the user. Each model variable and embedding vector has a corresponding PS node. Thus, every PS node has a subset of model variables and embedding tables.

The master will monitor PS nodes status similar to what it does for workers. In case a PS node fails, the master will try to relaunch it. Since PS nodes have higher priority than workers, if there are still some running workers, the relaunch will succeed by using either idle or preempted Kubernetes resources. If the relaunch fails, there are no workers left. The Elastic job has to wait for resources for the PS node and workers.

Each worker has a local copy of the model variables. After the master relaunches a PS node, the PS node can recover model variables from workers. For embedding vectors, PS must create replicas to support fault tolerance. For each PS node *PS(i)*, it will store *M* replicas in the following *M* PS nodes from *PS(i+1 % N)* to *PS(i+M % N)*. The relaunched PS node can recover embedding vectors from one of its replicas. If there are more than *M* continuously-indexed PS nodes failing, at least one PS node fails with all of its replicas. The ElasticDL job has to recover from a recent checkpoint.

## PS

### Compoments

![pserver](./images/pserver.png)

PS contains two main compoments:

- KVStore
- Optimizer

The worker initializes a model, and pushes parameters to KVStore. Before each step of training, the worker pulls the latest model from KVStore. After a round of forward/backward computation, the worker pushes gradients to the PS waiting for processing. Then, optimizer of PS will look up the corresponding parameter from KVStore. At last, it applies gradients to parameters, and updates parameter back to KVStore.

### Tensor Data Structure

To support data communication between pods, we introduce a `Tensor` proto message:

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
    repeated int64 dims = 3;
    bytes content = 4;
    repeated int64 indices = 5;
}
```

Correspondingly, we have a `Tensor` Python class.


```python
class Tensor(object):
    def __init__(self, name=None, value=None, indices=None):
        self.name = name
        self.value = value
        self.indices = indices
```

There are also some helper functions:

```python
def serialize_to_pb(tensor, pb):
    pass

def deserialize_from_pb(pb, tensor):
    pass

def convert_to_tf_variable(tensor):
    pass

def convert_to_tf_tensor(tensor):
    pass
```

### KVStore

For a common model variable, we use save it as a `tf.Variable` in Parameter DB. For the embedding table, we introduce a customized data structure.


```python
class EmbeddingTable(object):
    def __init__(self, name, meta_info):
        self.name = name
        self.meta_info = meta_info
        self.vectors = {}
        
    def get(self, indices):
        pass
        
    def set(self, indices, value):
        pass        
```

The name of embedding table is the embedding layer name. EmbeddingTable uses a dictionary `vectors` to store `<id, embedding_vector>` pairs.

The KVStore could be like following:

```python
class KVStore(object):
    def __init__(self):
        self.var_db = {}
        self.embedding_table_db = {}
        
    def get_param(self, name):
        pass
        
    def get_embedding_vector(self, name, indices):
        pass
        
    def set_param(self, name, value):
        pass
        
    def set_embedding_vector(self, name, indices, value):
        pass
```

### Optimizer

Once optimizer gets a gradient it will query the KVStore to get the corresponding parameter. Then it will apply the gradient to the parameter. It has a `tf.keras.optimizer` instance inside.

Many `tf.keras.optimizer` subclasses, such as `Adam` and `Adagrad` allocate and manage additional variables associated with the variables to train.  These are called `Slots`.

Embedding table slots are stored at KVStore, and other common parameter slots are stored and managed by `tf.keras.optimizer`.

The embedding table slot is also a embedding table data structure. For example, a embedding table parameter with name `embedding_layer0`, we will create a corresponding `embedding_layer0-momentum` EmbeddingTable object in `KVStore.embedding_table_db`.

We support async-SGD and sync-SGD both.

There are two ways to support async-SGD:
 
- Calling `apply_gradient` inside `push_gradient` gRPC service.
- Putting gradients into a gradient queue, and optimizer gets gradients from the queue immediately to `apply_gradient`.

In the first way, there may be several gRPC threads running in parallel. This will introduce race condition on parameter updating, some gradients may be overwrited.

The second way ensure each gradient could be applied, and decoupling these two procedures, `push_gradient` of worker and `apply_gradient` of optimizer. But the second way introduces more staleness in updating model, and may influence the final training accuracy.

We may consider the second way later.

In sync-SGD, optimizer needs to wait for a certain number of gradients, and then get the gradient after addition. We could implement a customized gradient queue structure to support such logic efficiently.

The interface of Gradient Queue could be like this:

```python
class GradientQueue(object):
    def __init__(self):
        self.grad_queue = queue.Queue()
    
    def get_gradient(self):
        pass
        
    def put_gradient(self):
        pass
```

### RPC Service

PServer provide RPC service for workers.

Since pserver will store a subset of the full model. And a worker will push/pull a submodel from the pserver. 

However, embedding table is initialized lazily in pserver, worker should also send embedding table information to pserver. We have to add another filed to describe embedding table related information.

The model message is defined as following:

```proto

message EmbeddingTableInfo{
    string name = 1;
    repeated int64 dims = 2;
    string initializer = 3;
}

message Model {
    int64 version = 1;
    repeated Tensor variables = 2;
    repeated EmbeddingTableInfo embedding_table_info = 3;
}
```

Model could also be used as gradients collection.


So the RPC service will be defined as following:

```proto

service PServer{
    rpc push_model(Model) returns (google.protobuf.Empty) {}
    rpc pull_variable(Model) returns (Model) {}
    rpc pull_embedding_vector(Tensor) returns (EmbeddingResponse) {}
    rpc push_gradient(Model) returns (google.protobuf.Empty)
}
```

The interfaces of PServer could be like this:


```python
class PServer(elasticdl_pb2_grpc.PServerServicer):
    def __init__(self, kvstore, grad_queue, opt):
        self.kvstore = KVStore()
        self.grad_queue = GradientQueue()
        self.opt = Optimizer(opt, self.kvstore, self.grad_queue)

    def push_model(self, request, _):
        pass

    def push_gradient(self, request, _):
        pass
        
    def pull_variable(self, request, _):
        pass

    # embedding param is handled lazily
    def pull_embedding_vector(self, reques, _):
        pass
```

### Checkpoint and Serving

Master will send signal to pservers to make checkpoint. Each pserver will save parameters in its current KVStore to a distributed file system.

Since a pserver only has a subset of the whole model, we have to merge these submodels to get final model for serving.


## Interactions among Master, PS and Worker
The following events involve interactions among the master, workers and PS:

* The master starts PS.
* Initialization of parameters in PS.
* Relaunch of PS.
* Workers get model variables from PS.
* Workers push gradients to PS.
* PS reports submodel version to the master.
* The master tells PS to save checkpoint.

### The master starts PS
When an ElasticDL task starts, `master.main` is responsible for starting PS as a Kubernetes service. Through kubernetes service, we can fix domain name for every PS node. 

After starting PS, `master.main` starts the master servicer and workers, and tells them the domain name of PS. For PS with embedding replicas, every PS node also needs to know the domain name of its replicas.

### Initialization of parameters in PS
PS does not have any model variable and model meta info after starting. Model meta info includes dimension of embedding layers, initialization methods of embedding vectors, initialization methods of slot variables in optimizer.

There are two ways for PS to get model variables and model meta info, one is to read from a checkpoint file, one is to obtain them from workers.

When `master.main` starts PS, `master.main` decides how to initialize PS. If `master.main` passes an argument specifying the checkpoint file name to PS, PS reads from the checkpoint. Otherwise, PS does nothing but waiting for the first `get_model` from worker. In the reponse of `get_model` call, PS tells the worker to initialize model, and report model variables and meta info to the PS.

Please Note that the worker only initializes model variables. ElasticDL adopts lazy initialization for embedding vectors. Please refer to "[Workers get model parameters from PS](#Workers-get-model-parameters-from-PS)" section.

### Relaunch of PS
In case a PS pod fails, the master will try to relaunch one PS and it should recover model variables and embedding tables.

For model variables, PS can recover from workers in the same way as the variable initialization.

For embedding tables, the `master.main` tells PS through in starting command that PS should recover from replica. If there is no replica, PS has to recover from checkpoint.

### Workers get model variables from PS
Before each forward-pass, workers need to get all model variables from PS. Currently, workers call function `get_model()` to get variables.

Workers get embedding vectors from PS when the forward-pass function of the ElasticDL embedding layer is called. PS may not possess all the embedding vectors needed because ElasticDL adopts lazy initialization for embedding vectors, i.e. iniatializing embedding vectors when they are needed. Thus, if a worker wants to pull some embedding vectors that are not existing in PS, PS will create and initialize these embedding vectors and return their value to the worker.

### Push Gradients
After backward-pass, workers push gradients to PS.

### PS reports submodel version to the master
The master needs to know the model version to decide when to save checkpoint and when to evaluate model. PS regularly reports the version of the submodel it possessed to the master. 

Please note different pserver has different submodel version. The master choose the maximum of these submodel versions as the current model version.

### The master tells PS to save checkpoint
When the master decides to save checkpoint, the master tells all the pservers to save checkpoint. Every pserver saves the submodel it possessed into a separate file.


## Embedding Replicas in PS
An ElasticDL job has *N* PS nodes. Embedding vectors are partitioned into these PS nodes. The user provides *M*, the number of replicas for embedding vectors. *M* must be smaller than *N* as each PS node uses other PS nodes to store its embedding replicas.

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

Each PS node has a thread dedicated for the replica synchronization:

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
