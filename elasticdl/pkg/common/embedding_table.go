package common

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
	newVector := NewEmbeddingVector(table.Dim)
	table.EmbeddingVector[index] = newVector
	return newVector
}

// GetEmbeddingVectors returns embedding vectors giving an array of indices
func (table *EmbeddingTable) GetEmbeddingVectors(indices []int64) []*Tensor {
	var vectors []*Tensor
	for _, index := range indices {
		vectors = append(vectors, table.GetEmbeddingVector(index))
	}
	return vectors
}
