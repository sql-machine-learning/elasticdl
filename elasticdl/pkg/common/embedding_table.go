package common

// #include "../c/vector.h"
import "C"
import "fmt"

// EmbeddingTable struct
type EmbeddingTable struct {
	Name            string
	Dim             int64
	Initializer     InitialzeFunc
	Dtype           DataType
	EmbeddingVector map[int64]*Vector
}

// NewEmbeddingTable creates an embedding table instance
func NewEmbeddingTable(name string, dim int64, initializer string, dtype DataType) *EmbeddingTable {
	var e EmbeddingTable = EmbeddingTable{
		Name:            name,
		Dim:             dim,
		EmbeddingVector: make(map[int64]*Vector),
		Dtype:           dtype,
	}
	// only support zero init now
	e.Initializer = ZeroInit(int(dim), dtype)
	return &e
}

// GetEmbeddingVector returns embedding vector giving an index
func (e *EmbeddingTable) GetEmbeddingVector(index int64) *Vector {
	if value, ok := e.EmbeddingVector[index]; ok {
		return value
	}
	newVector := NewEmptyVector(int(e.Dim), e.Dtype, e.Initializer)
	e.EmbeddingVector[index] = newVector
	return newVector
}

// GetEmbeddingVectors returns embedding vectors giving an array of indices
func (e *EmbeddingTable) GetEmbeddingVectors(indices ...int64) *Tensor {
	t := NewEmptyTensor(e.Name, e.Dtype, int64(len(indices)), e.Dim)
	t.Indices = indices
	for i, index := range indices {
		t.SetRow(i, e.GetEmbeddingVector(index))
	}
	return t
}

// SetEmbeddingVectors sets (indices, value) pair to embedding vector
func (e *EmbeddingTable) SetEmbeddingVectors(t *Tensor) error {
	if t.Dim.Len() != 2 || t.Dim[1] != e.Dim {
		return fmt.Errorf("unmatched dimension")
	}
	for i, index := range t.Indices {
		e.EmbeddingVector[index] = t.Row(i)
	}
	return nil
}
