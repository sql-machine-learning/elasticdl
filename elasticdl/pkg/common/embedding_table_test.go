package common

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestEmbeddingTableInit(t *testing.T) {
	e1 := NewEmbeddingTable("e1", 2, "zero")
	v1 := e1.GetEmbeddingVector(10)
	assert.Contains(t, e1.EmbeddingVector, int64(10))
	assert.Equal(t, len(v1.Value), 2)
	assert.Equal(t, v1.Value[0], float32(0.0))
}

func TestEmbeddingTableGet(t *testing.T) {
	e1 := NewEmbeddingTable("e1", 2, "zero")
	v1 := e1.GetEmbeddingVector(1)
	assert.Equal(t, len(v1.Value), 2)
	v1.Value[0] = 1.0
	v1.Value[1] = 2.0
	assert.Equal(t, len(e1.EmbeddingVector[1].Value), 2)
	assert.Equal(t, e1.EmbeddingVector[1].Value[0], float32(1.0))
	assert.Equal(t, e1.EmbeddingVector[1].Value[1], float32(2.0))

	indices := []int64{1, 3, 5, 7, 9}
	v := e1.GetEmbeddingVectors(indices)
	assert.Equal(t, v.Value[0], float32(1.0))
	assert.Equal(t, v.Value[1], float32(2.0))
	assert.Equal(t, v.Value[2], float32(0.0))
}

func TestEmbeddingTableSet(t *testing.T) {
	e := NewEmbeddingTable("e1", 2, "zero")
	i := []int64{1, 3, 5}
	v := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

	err := e.SetEmbeddingVectors(i, v)
	assert.Nil(t, err)

	v1 := e.GetEmbeddingVector(1)
	assert.True(t, CompareFloatArray([]float32{1.0, 2.0}, v1.Value, 0.0001))

	v3 := e.GetEmbeddingVector(3)
	assert.True(t, CompareFloatArray([]float32{3.0, 4.0}, v3.Value, 0.0001))

	v5 := e.GetEmbeddingVector(5)
	assert.True(t, CompareFloatArray([]float32{5.0, 6.0}, v5.Value, 0.0001))
}
