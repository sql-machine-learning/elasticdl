# Design Doc: Distributed Embedding Layer
## Motivation
Embedding layers are commonly used in deep learning to represent discrete variables, e.g., words, as continuous vectors, e.g., word embedding vectors. The parameter of an embedding layer, known as an *embedding table*, is a VxN-tensor, where V is the vocabulary size, and N is the output dimension or the dimension of the word embedding vectors. With a large V, the embedding table might out-size the memory, and we'd need model parallelism with distributed training.

TensorFlow 1.x has a native solution for distributed training. It starts multiple processes, each running the TensorFlow runtime and communicating with each other to form a distributed runtime. TensorFlow 1.x represents deep learning computations as a data structure known as *graphs*. The native distributed training strategies partition a graph into smaller ones, and each process executes a sub-graph. With low-level APIs like `tf.create_partitioned_variable`, the distributed runtime can split a large embedding table and save the pieces on various computers.

TensorFlow 2.x, known for the new eager-execution mode, does no longer rely on graphs. The API is more flexible than the graph-based API and allows us to implement distributed training out of the runtime. ElasticDL explores an alternative approach to model parallelism -- saving large tensors in an external distributed storage service, for example, Redis and Memcached.

## Distributed Storage

An embedding table is a data structure that maps a discrete value, i, to a vector, r.  Please be aware that the discrete value i might not be an integer.  For example, it could be a string representing the kind of a fruit.  And even if it is an integer, i might not be zero-based, consider an integer of year.  This property inspires us to save large embedding tables in distributed caching/storage services like Redis or Memcached that supports `get(i)` and `set(i, r)`.

We can run a global service to serve all deep learning jobs on a cluster, or one service for each job.

## Read, Init, and Update

In the forward-pass of each iteration, workers read the embedding table. Suppose that we implement the distributed embedding layer as a Keras layer class, the forward-pass computation is in the overloaded method `call`, which takes a minibatch of inputs and returns the corresponding outputs.

Suppose that a minibatch of training instances contains M unique discrete values, {iⱼ}, where j∈[0, M), we prefer the reading operation return M embedding vectors {rⱼ}.  

If an rⱼ doesn't exist, the `call` method must randomly initialize it by calling `set`. The on-the-fly initialization strategy doesn't create embedding vectors for unseen discrete IDs and works with batch and online learning.

In the configuration of asynchronous distributed SGD, each worker process maintains its local copy of the model, and the parameter server process has the global model.  As long as each process runs the training code in a thread, this configuration doesn't require thread-safe `get` and `set`.

In the synchronous configuration, all workers must use the same model in each iteration. More than one workers might initialize the same embedding vector simultaneously, thus requires a new atomic operation: `get_or_initialize` or `set_if_not_exists`.  Redis API provides `set_if_not_exists`.

Another case of calling `set` is model update. With either synchronous and asynchronous case, we can restrict that only worker 0 or the parameter server can update the embedding table. Thus it poses no requirement of thread-safe `set`.

## Related Work

Before introducing our distributed embedding layer, let us review that of TensorFlow as an inspiration.

### Embedding Layers in TensorFlow

TensorFlow assumes that an embedding table is a dense tensor, which implies that users must make sure that the discrete input i is a zero-based integer.  TensorFlow provides the [feature column API](https://www.tensorflow.org/guide/feature_columns#feature_columns_2) to convert strings and other features into zero-based integers. 

By calling `tf.create_partitioned_variable` to create the embedding table, users can create distributed embedding layers. TensorFlow provides two operators to lookup a distributed embedding table:

1. [`tf.nn.embedding_lookup`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/embedding_lookup), and
1. [`tf.nn.embedding_lookup_sparse`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/embedding_lookup_sparse).

We will dig into these two functions later in this document.  The Keras layer [`tf.keras.layers.Embedding`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/Embedding?hl=en) is a wrapper of `tf.nn.embedding_lookup`.

### `tf.keras.layers.Embedding` and `tf.nn.embedding_lookup`

The constructor of `tf.keras.layers.Embedding` is as follows:

```python
def __init__(
    input_dim,
    output_dim,
    embeddings_initializer='uniform',
    embeddings_regularizer=None,
    activity_regularizer=None,
    embeddings_constraint=None,
    mask_zero=False,
    input_length=None,
    **kwargs
)
```

It takes two required arguments, where `input_dim` is the maximum value of the input discrete value i, `output_dim` is the length of each embedding vector. This constructor creates an embedding layer and its parameter, the embedding table, as a (partitioned) tensor of shape `input_dim x output_dim`.

The method [`tf.keras.layers.Embedding.call`](https://github.com/tensorflow/tensorflow/blob/1f61f13f8715dc26dabe46a2686216674026d812/tensorflow/python/keras/layers/embeddings.py#L179) defines the forward-pass.  It simply calls `tf.nn.embedding_lookup`.

```python
  def call(self, inputs):
    dtype = K.dtype(inputs)
    if dtype != 'int32' and dtype != 'int64':
      inputs = math_ops.cast(inputs, 'int32')
    out = embedding_ops.embedding_lookup(self.embeddings, inputs)
    return out
```

The operator `tf.nn.embedding_lookup` has the following signature:

```python
tf.nn.embedding_lookup(
    params,
    ids,
    max_norm=None,
    name=None
)
```

The input `ids` is a dense tensor of int32 or int64 elements, where each element identifies an embedding vector in table `params`.  The output consists of embedding vectors shaped the same as `ids`.

Suppose that a minibatch has N data instances and each instance has M discrete values, we can form `ids` as a dense tensor of N x M.  The output of `tf.nn.embedding_lookup` has the shape N x M x O, where O is the length of a embedding vector, or `output_dim`.

### `tf.nn.embedding_lookup_sparse`

If the input is a sparse tensor, we can use [`tf.nn.embedding_lookup_sparse`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/embedding_lookup_sparse) instead.

```python
tf.nn.embedding_lookup_sparse(
    params,
    sp_ids,
    sp_weights,
    combiner=None,
    max_norm=None,
    name=None
)
```

The input sparse tensor `sp_ids` is a N x M SparseTensor of int64 ids where N is typically batch size and M is arbitrary. 
There must be at least one element in each row of `sp_ids`.

For each row in the dense tensor represented by sp_ids, the op looks up the embeddings for all ids in that row, multiplies them by the corresponding weight, and combines these embeddings as specified.  The `combiner` could be "mean", "sqrtn" or "sum". "sum" computes the weighted sum of the embedding results for each row. "mean" is the weighted sum divided by the total weight. "sqrtn" is the weighted sum divided by the square root of the sum of the squares of the weights.

Thus, the output from `tf.nn.embedding_lookup_sparse` is a dense tensor of the shape N x O.

## elasticdl.layers.Embedding

We plan to support both the fixed number of input ids as [tf.keras.layers.Embedding](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/Embedding?hl=en) and [tf.nn.embedding_lookup](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/embedding_lookup),  and inputs with varying number of ids as [tf.nn.embedding\_lookup\_sparse](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/embedding_lookup_sparse).

```
__init__(
    output_dim,
    embeddings_initializer='uniform',
    mask_zero=False,
    input_length=None,
    combiner=None,
)
```

Because the embedding table size is not fixed in advance, `input_dim` argument in [tf.keras.layers.Embedding](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/Embedding?hl=en) is not used by `elasticdl.layers.embedding`.

We also need to investigate if elasticdl.layers.embedding can support`embeddings_regularizer`, `activity_regularizer` and `embeddings_constraint`.


In `elasticdl.layers.embedding.call(inputs)`:

* if `inputs` is a N x M SparseTensor, `combiner` cannot be None. The output shape is `N x output_dim`.
* If `inputs` is a N x M dense tensor, `combiner` can be None or any of the supported reduction op.
    * If it is None, the output shape is `N x M x output_dim`.
    * If it is not None, the output shape is `N x output_dim`. 

In this way, elasticdl.layers.embedding supports ops as:

1. [tf.keras.layers.Embedding](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/Embedding?hl=en)
2. [tf.nn.embedding\_lookup\_sparse](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/embedding_lookup_sparse)
3. [tf.keras.layers.Embedding](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/Embedding?hl=en) + reduction op `combiner`.

In this design document, we will describe how to implement `elasticdl.layers.Embedding` with a dense tensor `inputs` and `combiner` as None. It can be extended to support SparseTensor `inputs` and/or `combiner` as a reduction op.

In the remaining of this document, we abbreviate `elasticdl.layers.Embedding` as `Embedding`.

## Distributed storage service for embedding table

### Master start distributed storage service when needed
Before starting workers, master should decide whether to start the distributed storage service. A Keras model, whether defined by functional API or by subclass, uses some pre-defined layers as its model building blocks. Master searches `Embedding` in the model layers. If found, master starts the distributed storage service, and passes the access point to the workers through WorkerManager.

```
// master.main
embedding_layers = find_layers(model, Embedding)
access_point = None
if embedding_layers:
    access_point = Embedding_service.start_embedding_service()
worker_manager = WorkerManager(
    ...
    embedding_access_point = access_point,
    ...
)
worker_manager.start_workers()

```

Distributed storage will be empty at first. We adopts lazy initialization for embedding vectors, i.e. embedding vectors will be created when they are needed.

### Workers supports `lookup_embedding` using access_point

Worker defines a [`lookup_embedding`](#pseudocode-for-lookup_embedding) function and the `Embedding` layer will use it in [`Embedding.call`](#pseudocode-for-Embeddingcall). `lookup_embedding` will use the access point to access the distributed storage.


## Design Proposal

### Forward-Pass
In the forward-pass of each iteration, embedding layer takes a minibatch of discrete ids and returns the corresponding embedding vectors. Here is a simple example:

```
Embedding Table with 3 (discrete id, embedding vector) pairs:
{
	0: [0, 1, 2, 3],
	1: [4, 5, 6, 7],
	2: [8, 9, 10, 11],
}

Embedding Layer Input:
a minibatch input with 3 instances, each with 2 discrete id
[
	[0, 2],
	[2, 2],
	[0, 1],
]

Embedding Layer Output:
a minibatch output with 3 instances, each with 2 embedding vectors
[
	[[0, 1, 2, 3], [8, 9, 10, 11]],
	[[8, 9, 10, 11], [8, 9, 10, 11]],
	[[0, 1, 2, 3], [4, 5, 6, 7]],
]
```
Below, we will illustrate the forward-pass of model with `Embedding` in detail.

In *ElasticDL*, the core function of model calculation is `worker.training_process_eagerly()`. It takes a minibatch of features and labels, performs forward calculation and backward calculation, and returns loss and gradients. Here is its code in [worker.py](../python/worker/worker.py):

```
[1]    def training_process_eagerly(self, features, labels):
[2]        # GradientTape is used for recording gradients
[3]        with tf.GradientTape() as tape:
[4]            outputs = self._model.call(features, training=True)
[5]            loss = self._loss(outputs, labels)
[6]            # Add regularization loss if any
[7]            if self._model.losses:
[8]                loss += tf.math.add_n(self._model.losses)
[9]       grads = tape.gradient(loss, self._model.trainable_variables)
[10]      return loss, grads
```

When a worker calls `model.call(features, training=True)` on line 4 in training with a minibatch, each *layer* in *model* will call `layer.call(inputs)` when it is its turn. 
For `Embedding.call(inputs)`, it will generate a list of unique ids from `inputs`, lookup corresponding embedding vectors to create a dense tensor *BET* (Batch Embedding Tensor) and assign the values of BET to `Embedding`'s output. Here is a simple example:

```
embedding table is E
minibatch inputs is [[2, 6], [9, 6]]

1. unique_ids is [2, 6, 9]

2. BET is a 2D tensor with 3 embedding vectors:
BET = [
	E[2], 
	E[6], 
	E[9]
]

3. output is a 3D tensor with 2 instances, each instance with 2 embedding vectors:
outputs[0][0] = BET[0] = E[2]
outputs[0][1] = BET[1] = E[6]
outputs[1][0] = BET[2] = E[9]
outputs[1][1] = BET[1] = E[6]
```

##### pseudocode for `Embedding.call`

```
def Embedding.call(self, inputs):
    unique_ids = get_unique_ids(inputs)
    
    # name is used for generating keys in external distributed storage
    # initializer is used for lazy initialization
    BET = self.worker.lookup_embedding(
    	unique_ids, self.name, self.embedding_initializer)
    	
    if self._tape:
        # In order to get the gradient of BET from automatic-differentiation,
        # worker should set Embedding._tape as the current
        # tf.GradientTape before Embedding.call() in training.
        self._tape.watch(BET)
        
        self._bet_list.append(BET)
        self._unique_ids_list.append(unique_ids)
        
    # assign BET values to outputs based on the ids:
    # outputs[i][j] = BET[index of inputs[i][j] value in unique_ids]
    outputs = assign_output(BET, unique_ids, inputs)
    return outputs
```
There are two things that require more explanation. 

First, model may call some embedding layers more than once during one forward-pass. Thus we use `list` to record `BET` and `unique_ids`. 

Second, worker should set `Embedding._tape` before `Embedding.call`. This piece of pseudocode will be put between line 3 and line 4 of `training_process_eagerly` code above:

```
for all embedding layers in worker:
	embedding.set_tape(tape)
```

#### Initialization of Embedding Vectors
As mentioned in above section, *ElasticDL* adopts lazy initialization for embedding vectors. We will create and initialize an embedding vector when a new id appears. Thus, when `Embedding.call()` calls `Worker.lookup_embedding()`, worker will do following things:

* get embedding vectors for known ids from external distributed storage 
* generate initial values for new ids, use `set_if_not_exists` to create them in the external distributed storage
* return all embedding vectors

##### pseudocode for `lookup_embedding`

```
def Worker.lookup_embedding(self, ids, initializer='uniform'):
    id_to_embedding, unknown_ids = self.embedding_service_gets(ids)
    if unknown_ids:
        embedding_vector_list = []
        for id in unknown_ids:
            embedding_vector_list.append((id, initialize_embedding(id, initializer))
        
        # atomic operation sets_if_not_exists guarantees correctness 
        # without multithread lock
        self.embedding_service_sets_if_not_exists(
            embedding_vector_list
        )
        id_to_embedding_new, unknown_ids_new = self.embedding_service_gets(unknown_ids)
        if unknown_ids_new:
            raise Error
        for k, v in id_to_embedding_new:
            id_to_embedding[k] = v
    return [id_to_embedding[i] for i in ids]
```

`initilizer` is a `string` and it is used for get initializer. For example, in *Keras* interface, `initilizer='random_normal'` indicates initializer `keras.initializers.RandomNormal()`.

### Backward-Pass
We get *BET_list* from all `Embedding` layers. Worker computes *BET_list*'s gradient *G_BET_list*, and reports *G_BET_list* to master. Then master updates the embedding vectors and writes back to the distributed storage.
#### Worker computes embedding vectors' gradient
Worker computes the gradients of the embedding vectors in the same way as the trainable variables' gradients. So we put them together by changing line 9 of `training_process_eagerly` to:

```
    grads = tape.gradient(loss, self._model.trainable_variables + BET_list)
    G_BET_list = grads[len(self._model.trainable_variables):]
```

#### Worker reports embedding vectors' gradient to master  

In the forward-pass, worker reads a set of embedding vectors corresponding to the input ids. In the backward-pass,
we only update these embedding vectors. Worker reports the embedding vectors' ids and gradients to master. Master can locate embedding vectors in embedding table with embedding vector ids and update corresponding embedding vectors. 
We put *G_BET* and *unique_ids* (get from Embedding layer instance) together to create a set of tensor slices ([tf.IndexedSlices](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/IndexedSlices)) as the embedding vector's gradient `G_E`:

```
G_E.values = G_BET_list
G_E.indices = unique_ids_list
```
#### Master update embedding table with embedding vectors' gradients

The master accumulates the embedding vectors' gradients `G_E` as `G_EM`. When there are enough gradients to update the model, the master updates embedding table `E` with the accumulated `G_EM`. In order to call `optimizer.apply_gradient` only once, we should concatenate the gradients of trainable variables `G_variable` and gradients `G_EM` together.

```
G_EM_grads_value_pair = []
for i, index in enumerated(G_EM.indices):
    embedding = embedding_lookup(index, embedding_name)
    G_EM_grads_value_pair.append((G_EM.values[i], embedding))

variable_grads_value_pair = list(zip(G_variable, model.trainable_variables))
updated_variables = optimizer.apply_gradient(
    variable_grads_value_pair + G_EM_grads_value_pair
)
updated_embeddings = updated_variables[len(G_variable):]
write_back_to_embedding_table(G_EM.indices, updated_embeddings)
```
#### Support slots in Optimizer

For [SGD](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/optimizers/SGD?hl=en), we can use `optimizer.apply_gradient` directly to update the embedding table as shown above.

Many other optimizers allocate and manage additional varaiables associated with the variables to train, which are called slots in TensorFlow. For example, [tf.keras.optimizers.Ftrl](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/optimizers/Ftrl?hl=en) has two slots (*accumulator* and *linear*). [tf.keras.optimizers.Adam](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/optimizers/Adam?hl=en) also has two slots (*m* and *v*).

For the optimizers with slots, we should also store these slots in the distributed storage together with their corresponding embedding tables. Also, we need to write a corresponding *embedding_optimizer* to use the slots stored in the distributed storage.  In  `embedding_optimizer.apply_gardient((G_EM.values[i], embedding))`, we need to get the corresponding slots for *G_EM*, and update them as well.


## Issues to solve

* How to checkpoint with EdlEmbedding layer?
* How to use the exact model version for evaluation in training?




