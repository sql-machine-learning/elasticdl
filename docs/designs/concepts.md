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

To update the model, workers compute and report gradients.  Accordingly, we have two kinds of gradients:

1. dense gradient, and
1. embedding table gradient

The content of dense gradient is the same as that of the dense parameter.  The content of embedding table gradient is the same as that of the embedding table.

On the parameter server, we'd prefer to maintain each embedding table as a map from ID to embedding vectors. With such a data structure, it is efficient to allocate memory for new embedding vectors. On the contrary, we'd concatenate embedding vectors into protobuf messages for parameter pulling and gradient pushing. We cannot use concatenated embedding vectors as the in-memory data structure on the PS, because allocating new embedding vectors involves resize the space of concatenated embedding vectors.

Let's make a short summary, following is all the core concepts of ElasticDL include:

- model = {dense parameter} + {embedding table}
- dense parameter = tensor + name
- embedding table = tensor + ID + name

## Message Representation

The definition of `elasticdl.proto`:

```proto
enum TensorElementType {
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
  TensorElementType dtype = 3;
}

message IndexedSlices {
  Tensor concat_embedding_vecs = 1;
  repeated int64 ids = 2;
}

message Model {
  int32 version = 1; // model updated times
  map<string, Tensor> dense_parameters = 2;
  map<string, IndexedSlices> indexed_slices = 3;
}
```

In memory data structure:

```go
import "elasticdl.org/elasticdl/pkg/proto"

type EmbeddingTable struct {
    Name            string
    Dim             int64
    Initializer     string
    EmbeddingVector map[int64]*proto.Tensor
}

type Model struct {
    Version           int32
    InitStatus        bool
    DenseParameters   map[string]*proto.Tensor
    EmbeddingTables   map[string]*EmbeddingTable
}
```

## RPC Service

Following is some auxiliary messages needed by RPC services.

```proto
message PullDenseParametersRequest {
  int32 version = 1;
}

message PullDenseParametersResponse {
  bool initialized = 1;
  map<string, Tensor> = 2;
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

message EmbeddingTableInfos {
  repeated EmbeddingTableInfo embedding_table_infos = 1
}

message PushGradientResponse {
  bool accepted = 1;
  int32 version = 2;
}
```

Following is RPC services between PS and worker.

```proto
service Pserver {
  rpc pull_dense_params(PullDenseParametersRequest) returns (PullDenseParametersResponse);
  rpc pull_embedding_table(PullEmbeddingTableRequest) returns (IndexedSlices);
  rpc push_dense_params(Model) returns (google.protobuf.Empty);
  rpc push_embedding_table_infos(EmbeddingTableInfos) returns (google.protobuf.Empty);
  rpc push_gradient(Model) returns (PushGradientResponse);
}
```
