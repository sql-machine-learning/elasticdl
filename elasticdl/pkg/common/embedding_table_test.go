package common

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestEmbeddingTableInit(t *testing.T) {
	e1 := NewEmbeddingTable("e1", 2, "zero", Float32)
	v1 := e1.GetEmbeddingVector(10)
	assert.Contains(t, e1.EmbeddingVector, int64(10))
	assert.Equal(t, v1.Length, 2)
	assert.Equal(t, v1.At(0), 0.0)
}

func TestEmbeddingTableGet(t *testing.T) {
	e1 := NewEmbeddingTable("e1", 2, "zero", Float32)
	v1 := e1.GetEmbeddingVector(1)
	assert.Equal(t, v1.Length, 2)
	v1.Set(0, float32(1))
	v1.Set(1, float32(2))
	assert.Equal(t, e1.EmbeddingVector[1].Length, 2)
	assert.Equal(t, e1.EmbeddingVector[1].At(0), 1.0)
	assert.Equal(t, e1.EmbeddingVector[1].At(1), 2.0)

	indices := []int64{1, 3, 5, 7, 9}
	tensor := e1.GetEmbeddingVectors(indices)
	tensorSlice := tensor.InplaceSlice().([]float32)
	assert.True(t, CompareFloatArray(tensorSlice, []float32{1, 2, 0, 0, 0, 0, 0, 0, 0, 0}, 0.00001))
}

func TestEmbeddingTableSet(t *testing.T) {
	e := NewEmbeddingTable("e1", 2, "zero", Float32)
	i := []int64{1, 3, 5}
	v := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

	tensor := NewTensor("e1", v, []int64{3, 2}, i)
	err := e.SetEmbeddingVectors(tensor)
	assert.Nil(t, err)

	v1 := e.GetEmbeddingVector(1)
	assert.True(t, CompareFloatArray([]float32{1.0, 2.0}, v1.InplaceSlice().([]float32), 0.0001))

	v3 := e.GetEmbeddingVector(3)
	assert.True(t, CompareFloatArray([]float32{3.0, 4.0}, v3.InplaceSlice().([]float32), 0.0001))

	v5 := e.GetEmbeddingVector(5)
	assert.True(t, CompareFloatArray([]float32{5.0, 6.0}, v5.InplaceSlice().([]float32), 0.0001))
}
