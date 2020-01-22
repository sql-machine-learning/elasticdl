package common

// EmbeddingTable struct
type EmbeddingTable struct {
	Name            string
	Dim             int64
	Initializer     string
	EmbeddingVector map[int64][]float32
}

// GetEmbeddingVector returns embedding vector giving an index
func (table *EmbeddingTable) GetEmbeddingVector(index int64) []float32 {
	if table.EmbeddingVector == nil {
		table.EmbeddingVector = make(map[int64][]float32)
	}
	if value, ok := table.EmbeddingVector[index]; ok {
		return value
	}
	// TODO(qijun) only support zero initializer now
	newVector := make([]float32, table.Dim)
	table.EmbeddingVector[index] = newVector
	return newVector
}

// GetEmbeddingVectors returns embedding vectors giving an array of indices
func (table *EmbeddingTable) GetEmbeddingVectors(indices []int64) [][]float32 {
	if table.EmbeddingVector == nil {
		table.EmbeddingVector = make(map[int64][]float32)
	}
	var vectors [][]float32
	for _, index := range indices {
		vectors = append(vectors, table.GetEmbeddingVector(index))
	}
	return vectors
}
