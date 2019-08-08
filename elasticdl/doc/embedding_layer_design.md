# ElasticDL Custom Embedding Layer Design
This document describes the design and the implementation of ElasticDL custom embedding layer.

## Symbols used in this document

`model`: the Keras model provided by the user for training

`minibatch_size`: the batch size used by each worker

`E[embedding_size][embedding_dim]`: an embedding table with `embedding_size` embedding vectors, each vector has `embedding_dim` values.

`BET`: Batch Embedding Tensor, a dense tensor containing all embedding vectors used by an embedding layer for processing one batch.

`optimizer`: an optimizer provided by the user for updating `model` using gradients.

## Embedding Layers

### Embedding layer with fixed input size
An embedding layer defines an embedding table `E[embedding_size][ embedding_dim]`. For an input containing a list of `n` ids `[id_i for i from 0 to n-1]`, the layer looks up the embedding vectors `E[id_i]`  and outputs a dense matrix `Output[n][embedding_dim]` in which `Output[i] = E[id_i]`.


```
class EdlEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 embedding_size,
                 embedding_dim,
                 name,
                 embedding_initializer="uniform",
                 )
```

Input shape:
2D integer tensor with shape `(minibatch_size, n)`

Output shape:
3D tensor with shape `(minibatch_size, n, embedding_dim)`.

Each data in a batch has a fixed number of ids `n` to lookup.

This embedding layer is similar to [tf.keras.layers.Embedding](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/Embedding?hl=en) and [tf.nn.embedding_lookup](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/embedding_lookup).


### Embedding layer with sparse lookup
If the numbers of ids in the input are not fixed, sparse lookup with a combiner can be used to reduce the varying number of embedding vectors into a tensor, similar to [tf.nn.embedding\_lookup\_sparse](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/embedding_lookup_sparse)

In this design document, we will describe how to implement `EdlEmbedding` with a fixed input size. It can be extended to support the sparse lookup and we will implement that in the future work.


## Implementation

### Step 1: [master] Embedding table initialization

The master searchs for `EdlEmbedding` layers in `model`. If found, layer attributes `(name, embedding_size, embedding_dim, embedding_initializer)` are used to initialize the embedding tables by creating a Redis cluster or memory blocks in the master. We implement the Redis cluster first as it support for arbitrary sizes of embedding tables.

```
embedding_layers = find_layer(model, EdlEmbedding)
if embedding_layers:
    embedding_layer_attributes = get_attibutes(embedding_layers)
    initialize_embedding_table(embedding_layer_attributes)
```



### Step 2: [worker] `EdlEmbedding.call(inputs)`

When a worker calls `model.call(features)` in training with a batch, each `layer` in `model` will call `layer.call(inputs)` when it is its turn. 
For `EdlEmbedding.call(inputs)`, it will generate a set of ids from its inputs, lookups the corresponding embedding vectors to create a dense tensor `BET` (Batch Embedding Tensor)  and assigns the layer output values from `BET`. Also, in order to get the gradient for `BET`, `BET` has to be added to the watchlist of the current `tf.GradientTape`.

```
def EdlEmbedding.call(self, inputs, training):
    id_set = get_id_set(inputs)
    BET = embedding_lookup(id_set, self.name)
    # tape is the current tf.GradientTape for automatic-differentiation.
    if training:
        tape.watch(BET)
    outputs = assign_output(BET, inputs)
    return outputs
```

For example:

```
embedding table is E
embedding_dim = 16
minibatch_size = 2
inputs = [[2, 6], [9, 6]]
id_set = [2, 6, 9]
BET shape is [3, 16], and BET[0] = E[2], BET[1] = E[6], BET[2] = E[9]
outputs shape is [2, 2, 16]:
outputs[0][0] = BET[0]
outputs[0][1] = BET[1]
outputs[1][0] = BET[2]
outputs[1][1] = BET[1]
```

For `embedding_lookup`, it can be implemented by a grpc call to the master. The master would lookup the embeddings from the embedding table. If Redis cluster is used, the worker may lookup from the Redis directly.

### Step 3: [worker] Report embedding gradient `G_E` to master

`BET`'s gradient `G_BET` is computed together with `model`'s variables. Together with `id_set` computed in `EdlEmbedding.call`, a set of tensor slices ([tf.IndexedSlices](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/IndexedSlices)) is created as the embedding's gradient `G_E`:

```
G_E.values = G_BET
G_E.indices = id_set
```
`G_E` is sent to the master together with all the gradients for the `model`'s variables in `ReportGradient` grpc call.

### Step 4: [master] Update embedding table with embedding gradients

The master accumulates the embedding gradients `G_E` as `G_EM`. When there are enough gradients to update the model, embedding table `E` is sparsely updated using the accumulated `G_EM`.

```
for i, index in enumerated(G_EM.indices):
    embedding = embedding_lookup(index, embedding_name)
    embedding = optimizer.apply_gradient((G_EM.values[i], embedding))
    write_back_to_embedding_table(embedding, index)
```

## Issues to solve
* How to checkpoint with EdlEmbedding layer?
* How to use the exact model version for evaluation in training?




