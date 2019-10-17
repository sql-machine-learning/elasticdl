# Parameter Server Design
This document describes the design of a distributed parameter server (PS) to support training scalability and PS fault tolerance in ElasticDL.


## Motivation
Parameter server based distributed training uses data parallelism to speed up training. PS stores model parameters. There are multiple workers that get model parameters from PS, compute gradients using different training data and send computed gradients to PS. PS iteratively updates these model parameters using gradients sent by workers. A PS based distributed training can use an arbitrary number of workers to support the scalability of training data size.

If a model has a very large size, it may not fit in the memory of a single parameter server. For example, many recommending and ranking models use embedding layers. The parameter of an embedding layer is an embedding table, which consists of multiple embedding vectors. When the number of embedding vectors is too large to store in a single PS, we need a distributed PS. A distributed PS contains multiple PS pods.  The distributed PS can partition model parameters and store different partitions in different PS pods. 

Furthermore, all workers need to get model parameters from PS and push computed gradients to PS. PS needs to process gradients from all workers to update model parameters. The bandwidth for data communication and the computation for gradient processing on PS are proportional to the number of workers and the model size. When the number of workers and/or the model size are large, a single PS can become a bottleneck in training due to the lack of enough bandwidth and gradient processing capacity. A distributed PS can distribute bandwidth and gradient processing into multiple PS pods to avoid becoming a training bottleneck.

Thus, a distributed PS can support the scalability of model size. it also supports the scalability of training data size by providing enough bandwidth and gradient processing capacity for large number of workers.

## PS Fault Tolerance
ElasticDL is a fault-tolerable deep learning system. A distributed PS consists of multiple PS pods. Each PS pod stores a partition of model parameters. Workers need to get model parameters from all PS pods in the forward-pass of each iteration. A failed PS pod will interrupt the training. We can relaunch any failed PS pod and recover the partition of model parameters on it to support PS fault tolerance.

In ElasticDL, a master is responsible for creating PS pods and worker pods using Kubernetes APIs. The master launches PS pods with high priority and launches worker pods with low priority. The master also monitor PS pods and worker pods status. In case a PS pod fails, the master will relaunch it using Kubernetes APIs. Since PS pods have a higher priority than worker pods, if there are still some running the worker pods, the relaunch will succeed by using either idle or preempted Kubernetes resources. If no worker pods left, ElasticDL has to wait for Kubernetes resources to continue the training.

After the relaunch of a PS pod, the PS pod needs to recover its partition of model parameters. For a worker, a minibatch of training data in a forward-pass contains some embedding vectors, which is a subset of embedding tables (the parameters of the corresponding embedding layers). The worker pulls all non-embedding parameters and a subset of embedding tables from PS pods in the training. The PS pod can recover non-embedding parameters from workers but not embedding tables.

In order to recover the embedding table partition in the relaunched PS pod, the distributed PS needs to have embedding table replicas. A PS pod can store its embedding table partition replicas in other PS pods. For example, assume there are *N* PS pods from *PS<sub>0</sub>* to *PS<sub>N-1</sub>*, *PS<sub>i</sub>* can stores its replica in *PS<sub>(i + 1) % N</sub>*. The relaunched PS pod  *PS<sub>i</sub>* can recover its embedding table partition from its replica in  *PS<sub>(i + 1) % N</sub>*.

## Model Parameter Partition
For a distributed PS with *N* PS pods, each PS pod stores a partition of model parameters.

For a non-embedding parameter, we use a hashing function *hash* and the parameter name *pname* to select a PS pod *PS<sub>i</sub>* where *i = hash(pname) % N* to store it. This hashing method can distribute non-embedding parameters into PS pods evenly in the number of parameters, but not the size of parameters. To distribute bandwidth and gradient processing more evenly among PS pods, we can consider a more intelligent method by taking parameter size into account in the future.

Each embedding layer has an embedding table which maps a discrete ID *i* to an embedding vector *v<sub>i</sub>*. Because many recommending and ranking models have large embedding tables, to support the training scalability as we discussed in [Motivation](#motivation), we partition each embedding table and store every partition in an unique PS pod. For an embedding vector *v<sub>i</sub>*, we select *PS<sub>i % N</sub>* to store it.

## Model Parameter Storage
Each PS node has a key-value store (KVStore) to store its partition of model parameters. We use a dictionary data structure for KVStore implementation.

PS is responsible to update model parameters using gradients sent by workers. We use [TensorFlow optimizers](https://www.tensorflow.org/api_docs/python/tf/optimizers) to apply gradients to model parameters as ElasticDL is based on TensorFlow. In order to update non-embedding parameters directly by TensorFlow optimizers, we choose to store each non-embedding parameter in the KVStore using the parameter name as its key, and a [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) instance as its value.

For an embedding table *ET* in an embedding layer *EL*, A worker needs to access a portion of embedding vectors *{v<sub>i</sub>}* in *ET* in each forward-pass iteration. A minibatch of training data in the iteration contains the corresponding discrete IDs *{i}* of *{v<sub>i</sub>}*. The Worker needs to pull these embedding vectors from their corresponding PS pods using the embedding layer name and the discrete IDs *{i}*. Thus, to store an embedding vector in the KVStore, we use its corresponding embedding layer name and discrete ID to form a pair, and use this pair as the key and the embedding vector itself as the value.

Each PS pod provides a RPC service `PServer` for workers to pull model parameters, with `pull_variable` for all non-embedding parameters and `pull_embedding_vector` for embedding vectors specified by an embedding layer name and a list of discrete ids.

```proto
service PServer{
    rpc pull_variable(PullModelRequest) returns (PullModelResponse);
    rpc pull_embedding_vector(Tensor) returns (Tensor);
}
```

## Model Parameter Initialization
We use lazy initialization strategy for model parameters in PS. Each PS pod has a parameter status. After the master launches a PS pod, the PS pod sets its parameter status as uninitialized. When a worker tries to get non-embedding parameters from the PS pod through RPC call `pull_variable`, the PS pod tells the worker that the parameter status is uninitialized in response. If the worker has already initialized non-embedding parameters, it sends non-embedding parameter values to the PS pod by a GRPC call `push_model`. If not, since the worker has the model definition, it can run a forward-pass using a training data minibatch to initialize non-embedding parameters first before `push_model`. 

```proto
service PServer{
    rpc push_model(Model) returns (google.protobuf.Empty);
}
```
When the PS pod receives non-embedding parameters in its first RPC service for `push_model`, it initialize non-embedding parameters and sets the parameter status as initialized.

For any embedding vector, the corresponding PS pod will initialize it in the first RPC service for `pull_embedding_vector` that contains this embedding vector. The PS pod needs the embedding vector size and the initialization method for the initialization. The embedding vector size and the initialization method are in the model definition and workers can send them in `push_model` together with non-embedding parameter values.

## Model Parameter Update
A worker computes gradients in each training iteration, which contain gradients for non-embedding parameters and some embedding vectors if the model contains embedding layers. The worker partitions these gradients using their corresponding parameter names or embedding layer names and discrete IDs for embedding vectors. Then the worker sends gradients partitions to their corresponding PS pods by RPC calls `push_gradient`.

```proto
service PServer{
    rpc push_gradient(PushGradientRequest) returns (PushGradientResponse);
}
```

When a PS pod receives gradients in `push_gradient`, it uses a TensorFlow optimizer to apply gradients to non-embedding parameters stored in its KVStore. 

We have already implemented an [`OptimizeWrapper`](https://github.com/sql-machine-learning/elasticdl/blob/develop/elasticdl/python/master/optimizer_wrapper.py) to sparsely update embedding vectors. `OptimizeWrapper` reads corresponding embedding vectors from the KVStore to form a temporary variable, uses a TensorFlow optimizer to apply gradients to this temporary variable, and writes results back to these embedding vectors in the KVStore. The PS pod can use this OptimizeWrapper directly to update embedding vectors.

In asynchronous SGD, the PS pod can apply gradients directly to update model parameters once it receives gradients. For synchronous SGD, the PS pod accumulates `grads_to_wait` gradients from workers then updates model parameters using these gradients. `grads_to_wait` is an ElasticDL argument specified by the user.

## Fixed Domain name for PS Pod
PS provides a RPC service for workers. Workers are using RPC stubs to send RPC service requests to PS. RPC stubs require PS pods domains. Because ElasticDL is Kubernetes-native, the master can use Kubernetes services to launch/relaunch PS pods with fixed domain names. In this way, workers do not need to re-configure RPC stubs after a PS pod relaunch.

## Model Parameter Recovery
The relaunched PS pod will recover model parameters to continue the training. 

For non-embedding parameters, the PS pod can recover them from workers in the same way as the parameter initialization by setting its parameter status as uninitialized. Workers will push non-embedding parameters to the PS pod through `push_model` after it detects the uninitialized parameter status in the response of `pull_variable`.

For embedding tables, PS creates replicas to support fault tolerance. For each PS pod *PS<sub>i</sub>*, it will store *M* replicas of its embedding table partitions in the following *M* PS pods from *PS<sub>i+1 % N</sub>* to *PS<sub>i+M % N</sub>*. The relaunched PS pod can recover embedding tables from one of its replicas. 

## Embedding Replica
Assume *E<sub>i</sub>* is the embedding table partition in PS pod *PS<sub>i</sub>*, it has *M* replicas which are stored in PS pods from *P<sub>(i + 1) % N</sub>* to *P<sub>(i + M) % N</sub>*. Also, *PS<sub>i</sub>* stores *M* other PS pod replicas *E<sub>(i - M) % N</sub>* to *E<sub>(i - 1) % N</sub>*. 

*PS<sub>i</sub>* maintains *M* updated embedding vector key sets *UKS_i(j) for j from 0 to M - 1*. When *PS<sub>i</sub>* sparsely updates its embedding table partition *E<sub>i</sub>*, it also add the updated embedding vector keys into these *M* sets. 


*PS<sub>i</sub>* also periodically synchronize the replicas stored in it from PS pods *PS<sub>(i - M) % N</sub>* to *PS<sub>(i - 1) % N</sub>*. The synchronization frequency can be several seconds.

Each PS pod will use a RPC call `SynchronizeEmbedding` to synchronize another PS pod replica stored in it, and use a RPC call `GetReplica` to get its replica from another PS pod for embedding vector recovery.

```
message SynchronizeEmbeddingRequest {
    int32 replica_index = 1;
}

message SynchronizeEmbeddingResponse {
    map<string, Tensor> embedding_vectors = 1;
}

service PServer{
    # RPC service for replica synchronization
    rpc SynchronizeEmbedding(SynchronizeEmbeddingRequest) returns (SynchronizeEmbeddingResponse);

    # RPC service for PS to recover embedding vectors after relaunch
    rpc GetReplica(SynchronizeEmbeddingRequest) returns (SynchronizeEmbeddingResponse);
}
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

The implementation of the RPC services:

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