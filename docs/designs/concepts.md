# Concepts Design Doc

This document describes core concepts of ElasticDL.

An ElasticDL model consists of two kinds of parameters:

1. dense parameters, and
1. embedding tables.

A dense parameter is a dense tensor with a name.  An embedding table is a map from some ID to embedding vectors.  An embedding table also has a name.  Formalizing the concepts, we have

- model = {dense parameter} + {embedding table}
- dense parameter = tensor + name
- embedding table = {id, tensor} + name

where the curly braces denote zero or more.

It is notable that in practice, we use the rule

- embedding table = tensor + index + name
- index = map[ID]address

because we append all embedding vectors into the tensor and use index, which is a map from ID to the starting address of the embedding vector in the tensor.

To update the model, workers compute and report gradients.  Accordingly, we have two kinds of gradients:

1. dense gradient, and
1. embedding table gradient

The content of dense gradient is the same as that of the dense parameter.  The content of embedding table gradient is the same as that of the embedding table.

Please be aware that if some embedding table maps zero-based successive ID values to embedding vectors.  Such kind of embedding table, if not too large, could be represented by a dense parameters.  In the rest of the design, the concept embedding table refers to a sparse/general embedding table, where the ID might not be zero-based successive values, and even not an numerical value.

Let's make a short summary, following is all the core concepts of ElasticDL include:

- model = {dense parameter} + {embedding table}
- dense parameter = tensor + name
- embedding table = tensor + index + name
- index = map[ID]address

## In-Memory Representation

EmbeddingTable is initialized lazily in PS. Worker sends item ids to request embedding vectors from PS. If the ID is not found in PS, the corresponding embedding vector will be initialized and sent back to the worker. In order to query and insert embedding vectors efficiently, we use a map data structure to represent EmbeddingTable.

```go
type TensorDtype = int64

const (
    Invalid TensorDtype = iota
    Int8 
    Int16
    Int32
    Int64
    Float16
    Float32
    Float64
    Bool
)

type Buffer struct {
    Data   []byte
    Length int64
    Dtype  TensorDtype
}

type Tensor struct {
    Content *Buffer
    Dims    []int64
    Indices []int64
}

type DenseParam struct {
    Name   string
    Tensor Tensor
}

type Emdedding = Buffer // alias for better understanding

type EmbeddingTable struct {
    Name        string
    Dim         int64
    Dtype       TensorDtype
    Initializer string
    Embeddings  map[int64]*Emdedding
}

type Model struct {
    Initialized     bool
    Version         int32
    Dtype           TensorDtype
    DenseParams     map[string]*DenseParam
    EmbeddingTables map[string]*EmbeddingTable
}
```


## Message Representation

```proto
enum TensorDtype {
  // Not a legal value for DataType. Used to indicate a DataType field
  // has not been set.
  DT_INVALID = 0;

  DT_INT8 = 1;
  DT_INT16 = 2;
  DT_INT32 = 3;
  DT_INT64 = 4;
  DT_FLOAT16 = 5;
  DT_FLOAT32 = 6;
  DT_FLOAT64 = 7;
  DT_BOOL = 8;
}

message Tensor {
  repeated int64 dims = 1;
  bytes content = 2;
  TensorDtype dtype = 3;
}

message DenseParam {
  string name = 1;
  Tensor tensor = 2;
}

message EmbeddingTable {
  string name = 1;
  Tensor tensor = 2;
  repeated int64 indices = 3;
}

message Model {
  int32 version = 1; // model updated times
  repeated DenseParam dense_params = 2;
  repeated EmbeddingTable embedding_tables = 3;
}
```

## RPC Service

Following is some auxiliary messages needed by RPC services.

```proto
message PullDenseParamsRequest {
  int32 version = 1;
}

message PullDenseParamsResponse {
  bool initialized = 1;
  repeated DenseParam dense_params = 2;
}

message PullEmbeddingTableRequest {
  string name = 1;
  repeated int64 indices = 2;
}

message EmbeddingTableInfo {
  string name = 1;
  int64 dim = 2;
  string initializer = 3;
}

message PushGradientResponse {
  bool accepted = 1;
  int32 version = 2;
}
```

Following is RPC services between PS and worker.

```proto
service Pserver {
  rpc pull_dense_params(PullDenseParamsRequest) returns (PullDenseParamsResponse);
  rpc pull_embedding_table(PullEmbeddingTableRequest) returns (EmbeddingTable);
  rpc push_dense_params(Model) returns (google.protobuf.Empty);
  rpc push_embedding_table_infos(EmbeddingTableInfo) returns (google.protobuf.Empty);
  rpc push_gradient(Model) returns (PushGradientResponse);
}
```
