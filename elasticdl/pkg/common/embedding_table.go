package common

import "fmt"

// EmbeddingTable struct
type EmbeddingTable struct {
	Name            string
	Dim             int64
	Initializer     string
	EmbeddingVector map[int64]*Vector
	Dtype           DataType
}

// NewEmbeddingTable creates an embedding table instance
func NewEmbeddingTable(name string, dim int64, initializer string, dtype DataType) *EmbeddingTable {
	var e EmbeddingTable
	e.Name = name
	e.Dim = dim
	e.Initializer = initializer
	e.EmbeddingVector = make(map[int64]*Vector)
	e.Dtype = dtype
	return &e
}

// GetEmbeddingVector returns embedding vector giving an index
func (e *EmbeddingTable) GetEmbeddingVector(index int64) *Vector {
	if value, ok := e.EmbeddingVector[index]; ok {
		return value
	}
	// TODO(qijun) only support zero initializer now
	vec := NewEmptyVector(int(e.Dim), e.Dtype)
	e.EmbeddingVector[index] = vec
	return vec
}

// GetEmbeddingVectors returns embedding vectors giving an array of indices
func (e *EmbeddingTable) GetEmbeddingVectors(indices []int64) *Tensor {
	dim := []int64{int64(len(indices)), e.Dim}
	t := NewEmptyTensor(e.Name, dim, e.Dtype)
	t.Indices = indices
	for i, index := range indices {
		t.SetRow(i, e.GetEmbeddingVector(index))
	}
	return t
}

// SetEmbeddingVectors sets Tensor to embedding vector
func (e *EmbeddingTable) SetEmbeddingVectors(t *Tensor) error {
	if len(t.Dim) != 2 || int64(e.Dim) != int64(t.Dim[1]) {
		return fmt.Errorf("Embedding vectors dim not match")
	}
	for i, index := range t.Indices {
		e.EmbeddingVector[index] = t.Row(i)
	}
	return nil
}
