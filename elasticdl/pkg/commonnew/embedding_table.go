package commonnew

import (
	"github.com/tensorflow/tensorflow/tensorflow/go/core/framework/types_go_proto"
)

// IndexedSlices is a wrapper for proto.IndexedSlices
type IndexedSlices struct {
	ConcatTensors *Tensor
	Ids           []int64
}

// EmbeddingTable struct
type EmbeddingTable struct {
	Dim              int64
	Initializer      string
	EmbeddingVectors map[int64]*Tensor
	Dtype            types_go_proto.DataType
}

// NewEmbeddingTable creates an embedding table instance
func NewEmbeddingTable(dim int64, initializer string, dtype types_go_proto.DataType) *EmbeddingTable {
	return &EmbeddingTable{
		Dim:              dim,
		Initializer:      initializer,
		EmbeddingVectors: make(map[int64]*Tensor),
		Dtype:            dtype,
	}
}

// GetEmbeddingVector returns an REFERENCE of embedding vector giving an index
func (e *EmbeddingTable) GetEmbeddingVector(index int64) *Tensor {
	if value, ok := e.EmbeddingVectors[index]; ok {
		return value
	}
	newVector := NewEmptyVector(e.Dim, e.Dtype)
	e.EmbeddingVectors[index] = newVector
	return newVector
}

// GetEmbeddingVectors returns COPYS of embedding vectors giving an array of indices
func (e *EmbeddingTable) GetEmbeddingVectors(indices []int64) *Tensor {
	dim := []int64{int64(len(indices)), e.Dim}
	tensor := NewEmptyTensor(dim, e.Dtype)
	for i, index := range indices {
		tensor.SetRow(int64(i), e.GetEmbeddingVector(index))
	}
	return tensor
}

// SetEmbeddingVectors sets (indices, value) pair to embedding vector
func (e *EmbeddingTable) SetEmbeddingVectors(idxslice *IndexedSlices) {
	for i, index := range idxslice.Ids {
		value := e.GetEmbeddingVector(index)
		copy(value.TensorContent, idxslice.ConcatTensors.GetRow(int64(i)).TensorContent)
	}
}
