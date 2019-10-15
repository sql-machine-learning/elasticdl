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

##PS Fault Tolerance

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
