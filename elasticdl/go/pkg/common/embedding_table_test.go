// Copyright 2020 The ElasticDL Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package common

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
	it := NewIndexedSlices(t1, []int64{1})
	e1.SetEmbeddingVectors(it)
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
	it := NewIndexedSlices(tensor, i)

	err := e.SetEmbeddingVectors(it)
	assert.Nil(t, err)

	v1 := e.GetEmbeddingVector(1)
	assert.True(t, CompareFloatArray([]float32{1.0, 2.0}, Slice(v1).([]float32), 0.0001), "SetEmbeddingVector FAIL")

	v3 := e.GetEmbeddingVector(3)
	assert.True(t, CompareFloatArray([]float32{3.0, 4.0}, Slice(v3).([]float32), 0.0001), "SetEmbeddingVector FAIL")

	v5 := e.GetEmbeddingVector(5)
	assert.True(t, CompareFloatArray([]float32{5.0, 6.0}, Slice(v5).([]float32), 0.0001), "SetEmbeddingVector FAIL")
}
