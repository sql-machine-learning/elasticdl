# ElasticDL Embedding Layer Design
This document describes the design of ElasticDL embedding layer.

## Terminology

*model*: the Keras model provided by the user for training

*minibatch_size*: the batch size used by each worker

*E*: an embedding table supports query with a key. The query result is an embedding vector which has *embedding_dim* values.

*BET*: Batch Embedding Tensor, a dense tensor containing all embedding vectors used by an embedding layer for processing one batch.

*optimizer*: an optimizer provided by the user for updating *model* using gradients.

## Embedding Layers

### Embedding layer with fixed input size
An embedding layer defines an embedding table *E*. For an input containing a list of *n* ids `[id_i for i from 0 to n-1]`, the layer looks up the embedding vectors *E[id_i]*  and outputs a dense matrix *Output\[n\]\[embedding_dim\]* and `Output[i] = E[id_i]`.


```
class EdlEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 embedding_dim,
                 embedding_initializer="uniform",
                 )
```

Input shape:
2D integer tensor with shape *(minibatch_size, n)*

Output shape:
3D tensor with shape *(minibatch_size, n, embedding_dim)*.

The number of ids in each data of a same batch is fixed to *n*.

This embedding layer is similar to [tf.keras.layers.Embedding](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/Embedding?hl=en) and [tf.nn.embedding_lookup](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/embedding_lookup).




### Embedding layer with sparse lookup
If the numbers of ids in the batch inputs are not fixed, sparse lookup with a combiner can be used to reduce the varying number of embedding vectors into a tensor, similar to [tf.nn.embedding\_lookup\_sparse](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/embedding_lookup_sparse)

In this design document, we will describe how to implement `EdlEmbedding` with a fixed input size. It can be extended to support the sparse lookup and we will implement that in the future.


## Implementation

### Step 1: Embedding table initialization

#### Step 1.1 Start Redis Service

There are two methods to start Redis service. We need to discuss them.

##### Method 1 `get_worker()`

In this method, all things will be done during the `init` process of `EdlEmbedding`. In the `init` process of `EdlEmbedding`, `EdlEmbedding` will first get worker handle through function `get_worker`, and worker will ask master to start the Redis service. Master will check the existence of Redis service. If not, master will start the Redis service. Then, master will return the access point of Redis service to worker.

In order to `get_worker()` in the keras layer `EdlEmbedding`, worker need to set itself to a global variable.

```
class EdlEmbedding:
    def __init__(self, **kwargs):
        self.worker = self.get_worker()
        self.worker.init_embedding_service()
				
        // standard keras layer init 
        super(EdlEmbedding, self).__init__(**kwargs)
				
    def get_worker(self):
        from elasticdl.python.common.worker_help import current_worker
        return current_worker
				
class Worker:
    def set_worker(self):
        from elasticdl.python.common.worker_help import current_worker
        current_worker = self
		
    def init_embedding_service(self):
        response = self.init_embedding_service_rpc()
        self.ip, self.port = response.ip, response.port

// RPC
def MasterServicer.StartEmbeddingServiceRPC(self, request):
    with self._embedding_service_lock:
        if not self._is_embedding_service_started:
            ip, port = self._start_embedding_service()
            self._is_embedding_service_started = True
    res = StartEmbeddingServiceResponse()
    res.ip, res.port = ip, port
    return res
```



##### Method 2 `find_layer()` and `set_worker()` 

In this method, the two things mentioned above will be done seperately. 

1. Master creates Redis service before starting worker. Master searchs for  `EdlEmbedding` layers. If found,  master starts a Redis service. 

2. Master start workers.

3. Workers search for `EdlEmbedding` layers. If found, workers pass itself to `EdlEmbedding`.

```
def EdlEmbedding.set_worker(self, worker):
    self.worker = worker

def Worker.__init__(self):
    embedding_layers = find_layers(model, EdlEmbedding)
    if embedding_layers:
        for layer in embedding_layers:
            layer.set_worker(self)
				
class MasterServicer.__init__(self, *args, **kwargs):
    ...
    // masterservicer init done
				
    embedding_layers = find_layers(model, EdlEmbedding)
    if embedding_layers:
        ip, port = self._start_embedding_service()

```



#### Step 1.2 Initialize embedding vectors

After model initialization, Redis is empty. Worker will create and initialize embedding vectors lazily. During the forward pass of model, `EdlEmbedding` will send `ids` to  worker, and worker should return every `id`'s embedding vector to `EdlEmbedding`.  Some ids are already in Redis and worker will get their embedding vectors from Redis. Worker will create and initialize embedding vectors for other ids. In order to avoid conflict between workers, worker will use interface `embedding_service_sets_if_not_exists` to report new embedding vectors.

`initilizer` is a `string` and it is used for get initializer. For example, in *keras* interface, `initilizer='random_normal'` indicates initializer `keras.initializers.RandomNormal()`.

```				
def Worker.embedding_lookup(self, ids, initializer='uniform'):
    id_to_embedding, unknown_ids = self.embedding_service_gets(ids)
    if unknown_ids:
        embedding_vector_list = []
        for id in unknown_ids:
            embedding_vector_list.append((id, initialize_embedding(id, initializer))
        
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


### Step 2: [worker] `EdlEmbedding.call(inputs)`
The training process in [worker.py](../python/worker/worker.py):

```
[1]   @tf.function
[2]   def training_process(self, features, labels):
[3]        with tf.GradientTape() as tape:
[4]            outputs = self._model.call(features, training=True)
[5]            loss = self._loss(outputs, labels)
[6]           # Add regularization loss if any
[7]            if self._model.losses:
[8]                loss += tf.math.add_n(self._model.losses)
[9]        grads = tape.gradient(loss, self._model.trainable_variables)
[10]       return loss, grads
```

When a worker calls `model.call(features, training=True)` on line 3 in training with a batch, each *layer* in *model* will call `layer.call(inputs)` when it is its turn. 
For `EdlEmbedding.call(inputs)`, it will generate a set of ids from its inputs, lookups the corresponding embedding vectors to create a dense tensor *BET* (Batch Embedding Tensor)  and assigns the value of BET to EdlEmbedding layer's output. Also, in order to get the gradient for *BET*, we add *BET* tensor to the watchlist of the current gradient tape. The gradient tape is updated in every batch training by add a function call between line 3 and 4 above:

```
    self.set_gradient_tape(tape)
```

Below is the pseudocode for `EdlEmbedding.call`:

```
def EdlEmbedding.call(self, inputs, training):
    unique_ids = get_unique_ids(inputs)
    BET = self.worker.embedding_lookup(
    	unique_ids, self.name, self.embedding_initializer)
    if training:
        # tape is the current tf.GradientTape for 
        # automatic-differentiation.
        tape = self.worker.get_gradient_tape()
        self.tape.watch(BET)
    # assign BET values to outputs based on the ids:
    # outputs[i][j] = BET[index of inputs[i][j] value in unique_ids]
    outputs = assign_output(BET, unique_ids, inputs)
    return outputs
```

For example:

```
embedding table is E
embedding_dim = 16
minibatch_size = 2
inputs = [[2, 6], [9, 6]]
unique_ids = [2, 6, 9]
BET shape is [3, 16], and BET[0] = E[2], BET[1] = E[6], BET[2] = E[9]
outputs shape is [2, 2, 16]:
outputs[0][0] = BET[0]
outputs[0][1] = BET[1]
outputs[1][0] = BET[2]
outputs[1][1] = BET[1]
```


### Step 3: [worker] Report embedding gradient(G_E) to master

We can compute *BET*'s gardient *G_BET* with other variables of *model* together by changing line 9 of `training_process` to:

```
    grads = tape.gradient(loss, self._model.trainable_variables + [BET])
    G_BET = grads[len(self._model.trainable_variables)]
```

Then put *G_BET* and *unique_ids* (computed in `EdlEmbedding.call`) together to create a set of tensor slices ([tf.IndexedSlices](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/IndexedSlices)) as the embedding's gradient `G_E`:

```
G_E.values = G_BET
G_E.indices = unique_ids
```

Send *G_E* and all other graidients of *model*'s variables to the master in `ReportGradient` grpc call.

### Step 4: [master] Update embedding table with embedding gradients

The master accumulates the embedding gradients `G_E` as `G_EM`. When there are enough gradients to update the model, the master updates embedding table `E` with the accumulated `G_EM`. In order to call `optimizer.apply_gradient` only once, we should concatenate the gradients of native tensorflow variable `G_origin` and gradients `G_EM` together. 

```
G_EM_grads_value_pair = []
for i, index in enumerated(G_EM.indices):
    embedding = embedding_lookup(index, embedding_name)
    G_EM_grads_value_pair.append((G_EM.values[i], embedding))

origin_grads_value_pair = list(zip(G_origin, model.trainable_variables))
updated_variables = optimizer.apply_gradient(
    origin_grads_value_pair + G_EM_grads_value_pair
)
updated_embeddings = updated_variables[len(G_origin):]
write_back_to_embedding_table(G_EM.indices, updated_embeddings)
```

#### Support slots in Optimizer

For [SGD](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/optimizers/SGD?hl=en), we can use `optimizer.apply_gradient` directly to update the embedding table as shown above.

Many other optimizers allocate and manage additional varaiables associated with the variables to train, which are called slots in TensorFlow. For example, [tf.keras.optimizers.Ftrl](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/optimizers/Ftrl?hl=en) has two slots (*accumulator* and *linear*). [tf.keras.optimizers.Adam](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/optimizers/Adam?hl=en) also has two slots (*m* and *v*). 

For the optimizers with slots, when the master adds an emdedding to Redis, it needs to add the corresponding slots. Also, we need to write a corresponding *embedding_optimizer*.  In  `embedding_optimizer.apply_gardient((G_EM.values[i], embedding))`, we need to get the corresponding slots for *G_EM*, and update the slots as well.

## Issues to solve

* How to checkpoint with EdlEmbedding layer?
* How to use the exact model version for evaluation in training?




