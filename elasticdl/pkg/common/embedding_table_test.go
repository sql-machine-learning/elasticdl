package common

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestEmbeddingTableInit(t *testing.T) {
	e1 := NewEmbeddingTable("e1", 2, "zero", Float32Dtype)
	v1 := e1.GetEmbeddingVector(10)
	assert.Equal(t, int(v1.Length), 2)
	assert.Equal(t, v1.At(0), float64(0.0))
}

func TestEmbeddingTableGet(t *testing.T) {
	e1 := NewEmbeddingTable("e1", 2, "zero", Float32Dtype)
	v1 := e1.GetEmbeddingVector(1)
	assert.Equal(t, int(v1.Length), 2)
	v1.Set(0, 1.0)
	v1.Set(1, 2.0)
	assert.Equal(t, int(e1.EmbeddingVector[1].Length), 2)
	assert.Equal(t, e1.EmbeddingVector[1].At(0), 1.0)
	assert.Equal(t, e1.EmbeddingVector[1].At(1), 2.0)

	indices := []int64{1, 3, 5, 7, 9}
	v := e1.GetEmbeddingVectors(indices...)
	assert.Equal(t, v.FlatAt(0), 1.0)
	assert.Equal(t, v.FlatAt(1), 2.0)
	assert.Equal(t, v.FlatAt(3), 0.0)
}

func TestEmbeddingTableSet(t *testing.T) {
	e := NewEmbeddingTable("e1", 2, "zero", Float32Dtype)
	i := []int64{1, 3, 5}
	tensor := NewTensor("", []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, []int64{3, 2}, i)

	e.SetEmbeddingVectors(tensor)

	v1 := e.GetEmbeddingVector(1)
	assert.True(t, CompareFloatArray([]float32{1.0, 2.0}, []float32{float32(v1.At(0)), float32(v1.At(1))}, 0.0001))

	v3 := e.GetEmbeddingVector(3)
	assert.True(t, CompareFloatArray([]float32{3.0, 4.0}, []float32{float32(v3.At(0)), float32(v3.At(1))}, 0.0001))

	v5 := e.GetEmbeddingVector(5)
	assert.True(t, CompareFloatArray([]float32{5.0, 6.0}, []float32{float32(v5.At(0)), float32(v5.At(1))}, 0.0001))
}
