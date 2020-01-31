package common

import "fmt"

// EmbeddingTable struct
type EmbeddingTable struct {
	Name            string
	Dim             int64
	Initializer     string
	EmbeddingVector map[int64]*Tensor
}

// NewEmbeddingTable creates an embedding table instance
func NewEmbeddingTable(name string, dim int64, initializer string) *EmbeddingTable {
	var table EmbeddingTable
	table.Name = name
	table.Dim = dim
	table.Initializer = initializer
	table.EmbeddingVector = make(map[int64]*Tensor)
	return &table
}

// GetEmbeddingVector returns embedding vector giving an index
func (table *EmbeddingTable) GetEmbeddingVector(index int64) *Tensor {
	if value, ok := table.EmbeddingVector[index]; ok {
		return value
	}
	// TODO(qijun) only support zero initializer now
	newVector := NewVector(table.Dim)
	table.EmbeddingVector[index] = newVector
	return newVector
}

// GetEmbeddingVectors returns embedding vectors giving an array of indices
func (table *EmbeddingTable) GetEmbeddingVectors(indices []int64) []float32 {
	var vectors []float32
	for _, index := range indices {
		vectors = append(vectors, table.GetEmbeddingVector(index)...)
	}
	return vectors
}

// SetEmbeddingVectors sets (indices, value) pair to embedding vector
func (table *EmbeddingTable) SetEmbeddingVectors(indices []int64, value []float32) error {
	if int64(len(indices))*table.Dim != int64(len(value)) {
		return fmt.Errorf("Embedding vectors dim not match")
	}
	for i, index := range indices {
		table.EmbeddingVector[index] = NewVector(table.Dim)
		start := int64(i) * table.Dim
		copy(table.EmbeddingVector[index].Value, value[start:start+table.Dim])
	}
	return nil
}
