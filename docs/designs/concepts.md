# Concepts Design Doc

This document describes core concepts used in ElasticDL, and discusses their representations.

## Core Concepts

Machine learning algorithms build a mathematical model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to perform the task.

There are usually two kinds of parameters in a model, one is dense parameter, the other is sparse parameter. The sparse parameter is used by the embedding layer, which is also called embedding table. The embedding table maps discrete ID *i* to embedding vector *vᵢ*.

Both dense parameter and embedding table includes a tensor data field, and some other auxiliary fields, like name. Embedding table has an extra auxiliary field called indices.

Tensor is used to represented a n-dim data. We support several data types, such as int8/int32/int64/float32/float64.

Besides, we have two kinds of gradients, dense gradient and embedding table gradient. The data structure of dense gradient is the same with dense parameter, and so is the embedding table gradient.

In ElasticDL, only the parameter of large embedding layer is represented by an embedding table. For small embedding layer, a dense parameter is used, and its gradient also has an indices field. In this case, the data structure of gradient is not dense param, but an embedding table.

Let's make a short summary, following is all the core concepts used in ElasticDL:

- Model
- DenseParam
- EmbeddingTable
- Tensor

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
