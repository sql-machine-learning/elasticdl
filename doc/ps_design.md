# ElasticDL Parameter Server Design Doc

## Overview
Currently, there is one parameter server (PS) co-existed with the master. In order to support multiple PSs and PS fault tolerance, we need to separate PS from the master. Besides a KV store for model variables and embedding tables, PS should also support model variable update and embedding table sparse update using gradients.

The master will create *N* PSs, with *N* as the number of PSs specified by the user. Each model variable and embedding vector has a corresponding PS. Thus, every PS has a subset of model variables and embedding tables.

The master will monitor PS pods status similar to what it does for worker pods. In case a PS pod fails, the master will try to relaunch it. Since PS pods have higher priority than worker pods, if there are still some running worker pods, the relaunch will succeed by using either idle or preempted Kubernetes resources. If the relaunch fails, there are no worker pods left. The Elastic job has to wait for resources for the PS pod and worker pods.

Each worker has a local copy of the model variables. After the master relaunches a PS, the PS pod can recover model variables from workers. For embedding vectors, PS must create replicas to support fault tolerance. For each PS *P(i)*, it will store *M* replicas in the following *M* PSs from *P(i+1 % N)* to *P(i+M % N)*. The relaunched PS can recover embedding vectors from one of its replicas. If there are more than *M* continuously-indexed PSs failing, at least one PS fails with all of its replica PSs. The ElasticDL job has to recover from a recent checkpoint.

## Parameter Server (PS)

## Interactions among Master, PS and Worker


## Embedding Replicas in PS
An ElasticDL job has *N* PSs. Embedding vectors are partitioned into these *N* PSs. The user provides *M*, the number of replicas for embedding vectors. *M* must be smaller than *N* as each PS uses other PSs to store its embedding replicas.

Assume *E(i)* is the embedding vectors in PS *PS(i)*, it has *M* replicas which are stored in PSs from *P(i + 1 % N)* to *P(i + M % N)*. Also, *PS(i)* has replicas for *E(i - M % N)* to *E(i - 1 % N)*. 

*PS(i)* stores *E(i)* as a dictionary so that every embedding vector has a corresponding key.  *PS(i)* maintains *M* updated embedding vector key sets *UKS_i(j) for j from 0 to M - 1*. When *PS(i)* sparsely updates its embedding vectors *E(i)*, it also add the updated embedding vector keys into these *M* sets. 


*PS(i)* also periodically synchronize the replicas stored in it from *E(i - M % N)* to *E(i - 1 % N)*. The synchronization frequency can be several seconds.

Each PS will provide a GRPC service for the replica synchronization.

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

Each PS has a thread dedicated for replica synchronization. In this dedicated thread, PS will synchronize the stored replicas in it.

```
# T is the number of seconds for synchronization frequency
# Assume current PS is PS(i), self._stub[index] is the stub for PS(i - index)'s GRPC server.
# self.replicas[index] is the replica for PS(i - index).
req = elasticdl_pb2.SynchronizeEmbeddingRequest()
while still training:
    time.sleep(T)
    for replica_index in range(M):
        req.replica_index = replica_index
        updated_vectors = self._stub[replica_index].SynchronizeEmbedding(req)
        for key in updated_vectors.embedding_vectors:
            self.replicas[index][key] = updated_vectors.embedding_vectors[key] 
```

The implementation of the GRPC services:

```
def SynchronizeEmbedding(self, request, _):
    synch_embeddings = elasticdl_pb2. SynchronizeEmbeddingResponse()
    # self.UKS are the M updated embedding vector key sets in current PS
    # self.embedding_vector are the embedding vectors in current PS
    with self.lock():
        for key in self.UKS[request.replica_index]:
            synch_embeddings.embedding_vectors[key].CopyFrom(self.embedding_vector[key])
        self.UKS.clear()
    return synch_embeddings
    
def GetReplica(self, request, _):
    replica = elasticdl_pb2. SynchronizeEmbeddingResponse()
    for key in self.replicas[request.replica_index]:
        replica.embedding_vectors[key].CopyFrom(self.replicas[request.replica_index][key])
    return replica
```
Note that PS also need the lock for adding updated embedding vector keys into `self.UKS` after embedding table sparse update.