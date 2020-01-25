package common

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestEmbeddingTableInit(t *testing.T) {
	e1 := NewEmbeddingTable("e1", 2, "zero")
	v1 := e1.GetEmbeddingVector(10)
	assert.Contains(t, e1.EmbeddingVector, int64(10))
	assert.Equal(t, len(v1), 2)
	assert.Equal(t, v1[0], float32(0.0))
}

func TestEmbeddingGet(t *testing.T) {
	e1 := NewEmbeddingTable("e1", 2, "zero")
	v1 := e1.GetEmbeddingVector(1)
	assert.Equal(t, len(v1), 2)
	v1[0] = 1.0
	v1[1] = 2.0
	assert.Equal(t, len(e1.EmbeddingVector[1]), 2)
	assert.Equal(t, e1.EmbeddingVector[1][0], float32(1.0))
	assert.Equal(t, e1.EmbeddingVector[1][1], float32(2.0))

	indices := []int64{1, 3, 5, 7, 9}
	vectors := e1.GetEmbeddingVectors(indices)
	assert.Equal(t, len(vectors), len(indices)*2)
	assert.Equal(t, vectors[0], float32(1.0))
	assert.Equal(t, vectors[1], float32(2.0))
	assert.Equal(t, vectors[2], float32(0.0))
}
