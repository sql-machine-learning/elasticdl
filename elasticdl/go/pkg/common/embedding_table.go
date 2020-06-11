// Copyright 2020 The SQLFlow Authors. All rights reserved.
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
	"github.com/tensorflow/tensorflow/tensorflow/go/core/framework/types_go_proto"
	"sync"
)

// EmbeddingTable struct
type EmbeddingTable struct {
	Dim              int64
	Initializer      string
	EmbeddingVectors map[int64]*Tensor
	Dtype            types_go_proto.DataType
	lock             sync.RWMutex
}

// NewEmbeddingTable creates an embedding table instance
func NewEmbeddingTable(dim int64, initializer string, dtype types_go_proto.DataType) *EmbeddingTable {
	return &EmbeddingTable{
		Dim:              dim,
		Initializer:      initializer,
		EmbeddingVectors: make(map[int64]*Tensor),
		Dtype:            dtype,
	}
}

// GetEmbeddingVector returns an REFERENCE of embedding vector giving an index
func (e *EmbeddingTable) GetEmbeddingVector(index int64) *Tensor {
	e.lock.RLock()
	if value, ok := e.EmbeddingVectors[index]; ok {
		e.lock.RUnlock()
		return value
	}
	e.lock.RUnlock()
	newVector := NewEmptyVector(e.Dim, e.Dtype)
	e.lock.Lock()
	// TODO(qijun) only support uniform initializer
	if e.Initializer == "uniform" {
		initializerFn := RandomUniform(-0.05, 0.05, int64(len(e.EmbeddingVectors)))
		initializerFn(newVector)
	}
	e.EmbeddingVectors[index] = newVector
	e.lock.Unlock()
	return newVector
}

// GetEmbeddingVectors returns COPYS of embedding vectors giving an array of indices
func (e *EmbeddingTable) GetEmbeddingVectors(indices []int64) *Tensor {
	dim := []int64{int64(len(indices)), e.Dim}
	tensor := NewEmptyTensor(dim, e.Dtype)
	for i, index := range indices {
		tensor.SetRow(int64(i), e.GetEmbeddingVector(index))
	}
	return tensor
}

// SetEmbeddingVectors sets (indices, value) pair to embedding vector
func (e *EmbeddingTable) SetEmbeddingVectors(idxslice *IndexedSlices) error {
	for i, index := range idxslice.Ids {
		value := e.GetEmbeddingVector(index)
		copy(value.Buffer, idxslice.ConcatTensors.GetRow(int64(i)).Buffer)
	}
	return nil
}

// ToIndexedSlices transforms embedding table format to indexed slices format
func (e *EmbeddingTable) ToIndexedSlices() *IndexedSlices {
	e.lock.RLock()
	ids := make([]int64, 0, len(e.EmbeddingVectors))
	for k := range e.EmbeddingVectors {
		ids = append(ids, k)
	}
	e.lock.RUnlock()
	return NewIndexedSlices(e.GetEmbeddingVectors(ids), ids)
}
