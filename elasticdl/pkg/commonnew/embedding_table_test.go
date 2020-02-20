package commonnew

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestEmbeddingTableInit(t *testing.T) {
	e1 := NewEmbeddingTable(2, "zero", Float32)
	v1 := e1.GetEmbeddingVector(10)
	assert.Contains(t, e1.EmbeddingVectors, int64(10))
	assert.Equal(t, Slice(v1).([]float32), []float32{0, 0}, "NewEmbeddingTable FAIL")
}

func TestEmbeddingTableGet(t *testing.T) {
	e1 := NewEmbeddingTable(2, "zero", Float32)
	v1 := e1.GetEmbeddingVector(1) // Note: this is a reference type, future changes have effect on it
	t1 := NewTensor([]float32{1, 2}, []int64{1, 2})
	var is1 IndexedSlices
	is1.ConcatTensors = t1
	is1.Ids = []int64{1}
	e1.SetEmbeddingVectors(&is1)
	assert.Equal(t, Slice(v1).([]float32), []float32{1, 2}, "GetEmbeddingVector FAIL")

	indices := []int64{1, 3, 5, 7, 9}
	v := e1.GetEmbeddingVectors(indices) // Note: this is a copy type
	assert.Equal(t, Slice(v).([]float32), []float32{1, 2, 0, 0, 0, 0, 0, 0, 0, 0}, "GetEmbeddingVectors FAIL")
}

func TestEmbeddingTableSet(t *testing.T) {
	e := NewEmbeddingTable(2, "zero", Float32)
	i := []int64{1, 3, 5}
	v := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	tensor := NewTensor(v, []int64{3, 2})
	var is IndexedSlices
	is.ConcatTensors = tensor
	is.Ids = i
	e.SetEmbeddingVectors(&is)

	v1 := e.GetEmbeddingVector(1)
	assert.True(t, CompareFloatArray([]float32{1.0, 2.0}, Slice(v1).([]float32), 0.0001), "SetEmbeddingVector FAIL")

	v3 := e.GetEmbeddingVector(3)
	assert.True(t, CompareFloatArray([]float32{3.0, 4.0}, Slice(v3).([]float32), 0.0001), "SetEmbeddingVector FAIL")

	v5 := e.GetEmbeddingVector(5)
	assert.True(t, CompareFloatArray([]float32{5.0, 6.0}, Slice(v5).([]float32), 0.0001), "SetEmbeddingVector FAIL")
}
