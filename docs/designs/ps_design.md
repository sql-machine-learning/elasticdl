# ElasticDL Parameter Server Design
This document describes the design of a distributed parameter server for ElasticDL.

## Motivation
Parameter server (PS) stores model parameters which are used by workers. Workers get model parameters from PS, compute gradients using different training data and send computed gradients to PS. PS iteratively updates these model parameters using gradients sent by workers. A PS based distributed training can use an arbitrary number of workers to support the scalability of training data size.

We want to have one or more PS instances in each ElasticDL job. One reason is that models could be large and overrun the memory space of a single PS instance. In such case, we need to partition the model and store different partitions in different PS instances. Even if the model is not too big and fits in the memory of a single PS instance, we might still want to partition the model, as this distributes the model parameter communication from workers among PS instances. This also distributes the computation on PS such as parameter optimization. 

## PS Fault Tolerance
ElasticDL is a Kubernetes-native fault-tolerable deep learning system. An ElasticDL distributed PS consists of multiple PS pods. Each PS pod stores a partition of model parameters. Workers need to get model parameters from all PS pods. A failed PS pod will interrupt the training. We can relaunch any failed PS pod and recover the corresponding model parameter partition to support PS fault tolerance.

In ElasticDL, a master is responsible for creating PS pods and worker pods using Kubernetes APIs. The master launches PS pods with high priority and launches worker pods with low priority. The master also monitor PS and worker pods status. In case a PS pod fails, the master will relaunch it using Kubernetes APIs. Since PS pods have a higher priority than worker pods, if there are still some worker pods running, the relaunch will succeed by using either idle or preempted Kubernetes resources. If no worker pods left, ElasticDL has to wait for Kubernetes resources to continue the training.

After the relaunch of a PS pod, the PS pod needs to recover its partition of model parameters. The model may contain one or more embedding layers with embedding tables as their parameters. If so, a minibatch of training data in a worker contains some embedding vectors, which is a subset of embedding tables. The worker pulls all non-embedding parameters and and only a subset of embedding tables from PS pods in the training. Thus, the PS pod can recover non-embedding parameters from workers but not embedding tables.

In order to recover the embedding table partition in the relaunched PS pod, PS needs to replicate embedding tables. A PS pod can store its embedding table partition replicas in other PS pods. For example, assume there are *N* PS pods from *PS<sub>0</sub>* to *PS<sub>N-1</sub>*, *PS<sub>i</sub>* can stores its replica in *PS<sub>(i + 1) % N</sub>*. The relaunched PS pod  *PS<sub>i</sub>* can recover its embedding table partition from its replica in  *PS<sub>(i + 1) % N</sub>*.

## Model Parameter Partition
For a distributed PS with *N* PS pods, each PS pod stores a model parameter partition.

For a non-embedding parameter, we store it in a PS pod *PSᵢ*. We select *PSᵢ* using a hashing function *hash* and the parameter name *pname*: 

```
i = hash(pname) % N
```
 This hashing method can distribute non-embedding parameters into PS pods evenly in the number of parameters, but not the size of parameters. To distribute communication and computation for gradient processing more evenly among PS pods, we can consider a more intelligent method by taking parameter size into account in the future.

Each embedding layer has an embedding table which maps a discrete ID *i* to an embedding vector *vᵢ*. Because many recommending and ranking models have large embedding tables, we partition each embedding table and store every partition in an unique PS pod. For an embedding vector *vᵢ*, we select *PS<sub>i % N</sub>* to store it.

## Model Parameter Storage
Each PS node has a dictionary data structure to store its partition of model parameters.

We choose to store each non-embedding parameter using the parameter name as its key, and a [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) instance as its value. This is because that we want to update non-embedding parameters directly by [TensorFlow optimizers](https://www.tensorflow.org/api_docs/python/tf/optimizers).

If a model has one or more embedding layers, a minibatch of training data contains a set of discrete IDs. These discrete IDs correspond to a set of embedding vectors. The Worker needs to pull these embedding vectors from their corresponding PS pods using the embedding layer name and the discrete IDs. To store an embedding vector, We use its corresponding embedding layer name and discrete ID to form a pair, and use this pair as the key and the embedding vector itself as the value.

Each PS pod provides RPC services for workers to pull model parameters. RPC service `pull_variable` is to pull all non-embedding parameters. RPC service `pull_embedding_vector` is to pull embedding vectors specified by an embedding layer name and a list of discrete IDs.

```proto
service PServer{
    rpc pull_variable(PullModelRequest) returns (PullModelResponse);
    rpc pull_embedding_vector(Tensor) returns (Tensor);
}
```

## Model Parameter Initialization
We use lazy initialization for model parameters in PS. Each PS pod has a parameter status, which is `uninitialized` after the PS pod launch. When a worker tries to get non-embedding parameters from the PS pod through a RPC call `pull_variable`, the PS pod tells the worker that the parameter status is `uninitialized` in response. If the worker has already initialized non-embedding parameters, it sends non-embedding parameter values to the PS pod by a GRPC call `push_model`. If not, since the worker has the model definition and some training data, it can run a forward-pass to initialize non-embedding parameters first before `push_model`. 

```proto
service PServer{
    rpc push_model(Model) returns (google.protobuf.Empty);
}
```
When the PS pod receives non-embedding parameters in its first RPC service for `push_model`, it initializes non-embedding parameters and sets the parameter status as `initialized`.

For an embedding vector, the corresponding PS pod will initialize it in the first `pull_embedding_vector` service that contains this embedding vector. The PS pod needs the embedding vector size and the initialization method for the initialization. The embedding vector size and the initialization method are in the model definition and workers can send them in `push_model` to PS pods together with non-embedding parameter values.

## Model Parameter Update
A worker computes gradients in each training iteration, which contain gradients for non-embedding parameters and some embedding vectors if applicable. The worker partitions these gradients using their corresponding parameter names or embedding layer names and discrete IDs for embedding vectors. Then the worker sends gradient partitions to their corresponding PS pods by RPC calls `push_gradient`.

```proto
service PServer{
    rpc push_gradient(PushGradientRequest) returns (PushGradientResponse);
}
```

When a PS pod receives gradients in `push_gradient`, it uses a TensorFlow optimizer to apply gradients to non-embedding parameters. 

We have already implemented an [`OptimizeWrapper`](https://github.com/sql-machine-learning/elasticdl/blob/develop/elasticdl/python/master/optimizer_wrapper.py) to sparsely update embedding vectors. `OptimizeWrapper` uses corresponding embedding vectors to form a temporary variable, applies gradients to this temporary variable, and writes results back to these embedding vectors. The PS pod can use this OptimizeWrapper directly to update embedding vectors.

In asynchronous SGD, the PS pod can apply gradients directly to model parameters once it receives gradients. For synchronous SGD, the PS pod accumulates `grads_to_wait` gradients from workers then updates model parameters with these gradients. `grads_to_wait` is an ElasticDL argument specified by the user.

## Fixed Domain name for PS Pod
Each PS pod provides RPC services for workers. Workers are using RPC stubs to send RPC service requests to PS pods. RPC stubs require PS pod domains. Because ElasticDL is Kubernetes-native, the master can use Kubernetes services to launch/relaunch PS pods with fixed domain names. In this way, workers do not need to re-configure RPC stubs after a PS pod relaunch.

## Model Parameter Recovery
The relaunched PS pod will recover model parameters to continue the training. 

For non-embedding parameters, the PS pod can recover them from workers in the same way as the parameter initialization by setting its parameter status as `uninitialized`.

For embedding tables, PS creates replicas to support fault tolerance. For each PS pod *PSᵢ*, it can store *M* replicas of its embedding table partitions in *M* PS pods from *PS<sub>i+1 % N</sub>* to *PS<sub>i+M % N</sub>*. The relaunched PS pod can recover embedding tables from one of its replicas. 

## Embedding Replica
Assume *Eᵢ* is the embedding table partition in PS pod *PSᵢ*, it has *M* replicas stored in PS pods from *P<sub>(i + 1) % N</sub>* to *P<sub>(i + M) % N</sub>*. Also, *PSᵢ* stores *M* other PS pod replicas *E<sub>(i - M) % N</sub>* to *E<sub>(i - 1) % N</sub>*. 

*PSᵢ* maintains *M* updated embedding vector key sets *UKSᵢ(j) for j ∈ [0， M)*. When *PSᵢ* sparsely updates its embedding table partition *Eᵢ*, it also adds the updated embedding vector keys into these *M* sets. 


*PSᵢ* also periodically synchronize the replicas stored in it from PS pods *PS<sub>(i - M) % N</sub>* to *PS<sub>(i - 1) % N</sub>*. The synchronization frequency can be several seconds.

*PSᵢ* uses *M* RPC calls `SynchronizeEmbedding` the replicas store in it. `replica_index` values in `SynchronizeEmbeddingRequest` are from *(i - M) % N* to *(i - 1) % N*.

When *PSᵢ* needs to recover its embedding vectors after relaunch, it chooses a pod *PSⱼ* from *P<sub>(i + 1) % N</sub>* to *P<sub>(i + M) % N</sub>* which is still alive. *PSᵢ* uses a RPC call `GetReplica` to get its replica from *PSⱼ*.


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