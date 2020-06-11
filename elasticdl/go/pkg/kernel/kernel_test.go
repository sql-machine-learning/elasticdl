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

package kernel

import (
	"math"
	"math/rand"
	"testing"

	"elasticdl.org/elasticdl/pkg/common"
	"github.com/stretchr/testify/assert"
)

func TestSGD(t *testing.T) {
	const size int = 10
	a := make([]float32, size)
	b := make([]float32, size)
	var lr float32 = 0.1

	for i := 0; i < size; i++ {
		a[i] = rand.Float32()
		b[i] = rand.Float32()
	}

	d := []int64{2, 5}
	grad := common.NewTensor(a, d)
	param := common.NewTensor(b, d)

	expected := make([]float32, size)
	for i := 0; i < size; i++ {
		expected[i] = b[i] - lr*a[i]
	}

	SGD(grad, param, lr)
	assert.Equal(t, expected, b)
}

func TestSparseSGD(t *testing.T) {
	a := []float32{-1.0, -1.0, -1.0, -1.0, -1.0, -1.0}
	d := []int64{3, 2}
	indices := []int64{1, 3, 3}
	grad := common.NewTensor(a, d)
	isgrad := common.NewIndexedSlices(grad, indices)

	table := common.NewEmbeddingTable(2, "zero", common.Float32)

	err := SparseSGD(isgrad, table, 0.1)
	assert.Nil(t, err)
	assert.Equal(t, 2, len(table.EmbeddingVectors))

	v1 := table.GetEmbeddingVector(1)
	assert.Equal(t, []float32{0.1, 0.1}, common.Slice(v1).([]float32))

	v3 := table.GetEmbeddingVector(3)
	assert.Equal(t, []float32{0.2, 0.2}, common.Slice(v3).([]float32))
}

func TestAdam(t *testing.T) {
	const size int = 10
	rawGrad := make([]float32, size)
	rawParam := make([]float32, size)
	rawM := make([]float32, size)
	rawV := make([]float32, size)
	dim := []int64{2, 5}

	for i := 0; i < size; i++ {
		rawGrad[i] = rand.Float32()
		rawParam[i] = rand.Float32()
		rawM[i] = rand.Float32()
		rawV[i] = rand.Float32()
	}

	grad := common.NewTensor(rawGrad, dim)
	param := common.NewTensor(rawParam, dim)
	m := common.NewTensor(rawM, dim)
	v := common.NewTensor(rawV, dim)

	var lr float32 = 0.1
	var step int64 = 5
	var beta1 float32 = 0.9
	var beta2 float32 = 0.999
	var epsilon float32 = 1e-8

	expectedParam := make([]float32, size)
	expectedM := make([]float32, size)
	expectedV := make([]float32, size)

	for i := 0; i < size; i++ {
		expectedM[i] = beta1*rawM[i] + (1-beta1)*rawGrad[i]
		expectedV[i] = beta2*rawV[i] + (1-beta2)*rawGrad[i]*rawGrad[i]
	}

	for i := 0; i < size; i++ {
		expectedParam[i] = rawParam[i] - lr*expectedM[i]/
			(1-float32(math.Pow(float64(beta1), float64(step))))/
			(float32(math.Sqrt(float64(expectedV[i]/
				(1-float32(math.Pow(float64(beta2), float64(step)))))))+
				epsilon)
	}

	Adam(grad, param, m, v, lr, step, beta1, beta2,
		epsilon, false, nil)

	assert.True(t, common.CompareFloatArray(expectedM, common.Slice(m).([]float32), 0.0001))
	assert.True(t, common.CompareFloatArray(expectedV, common.Slice(v).([]float32), 0.00001))
	assert.True(t, common.CompareFloatArray(expectedParam, common.Slice(param).([]float32), 0.00001))
}

func TestAdamWithAmsgrad(t *testing.T) {
	const size int = 10
	rawGrad := make([]float32, size)
	rawParam := make([]float32, size)
	rawM := make([]float32, size)
	rawV := make([]float32, size)
	rawMaxSquare := make([]float32, size)
	dim := []int64{2, 5}

	for i := 0; i < size; i++ {
		rawGrad[i] = rand.Float32()
		rawParam[i] = rand.Float32()
		rawM[i] = rand.Float32()
		rawV[i] = rand.Float32()
		rawMaxSquare[i] = rand.Float32()
	}

	grad := common.NewTensor(rawGrad, dim)
	param := common.NewTensor(rawParam, dim)
	m := common.NewTensor(rawM, dim)
	v := common.NewTensor(rawV, dim)
	maxSquare := common.NewTensor(rawMaxSquare, dim)

	var lr float32 = 0.1
	var step int64 = 5
	var beta1 float32 = 0.9
	var beta2 float32 = 0.999
	var epsilon float32 = 1e-8

	expectedParam := make([]float32, size)
	expectedM := make([]float32, size)
	expectedV := make([]float32, size)
	expectedMaxSquare := make([]float32, size)

	for i := 0; i < size; i++ {
		expectedM[i] = beta1*rawM[i] + (1-beta1)*rawGrad[i]
		expectedV[i] = beta2*rawV[i] + (1-beta2)*rawGrad[i]*rawGrad[i]
		if rawMaxSquare[i] < expectedV[i] {
			expectedMaxSquare[i] = expectedV[i]
		} else {
			expectedMaxSquare[i] = rawMaxSquare[i]
		}

	}

	for i := 0; i < size; i++ {
		expectedParam[i] = rawParam[i] - lr*expectedM[i]/
			(1-float32(math.Pow(float64(beta1), float64(step))))/
			(float32(math.Sqrt(float64(expectedMaxSquare[i]/
				(1-float32(math.Pow(float64(beta2), float64(step)))))))+
				epsilon)
	}

	Adam(grad, param, m, v, lr, step, beta1, beta2,
		epsilon, true, maxSquare)

	assert.True(t, common.CompareFloatArray(expectedM, common.Slice(m).([]float32), 0.00001))
	assert.True(t, common.CompareFloatArray(expectedV, common.Slice(v).([]float32), 0.00001))
	assert.True(t, common.CompareFloatArray(expectedParam, common.Slice(param).([]float32), 0.00001))
	assert.True(t, common.CompareFloatArray(expectedMaxSquare, common.Slice(maxSquare).([]float32), 0.00001))
}

func TestSparseAdam(t *testing.T) {
	const size int = 10
	rawGrad := make([]float32, size)
	rawParam := make([]float32, size)
	rawM := make([]float32, size)
	rawV := make([]float32, size)
	rawMaxSquare := make([]float32, size)
	dim := []int64{1, 10}

	var embdim int64 = 10
	ptable := common.NewEmbeddingTable(embdim, "zero", common.Float32)
	mtable := common.NewEmbeddingTable(embdim, "zero", common.Float32)
	vtable := common.NewEmbeddingTable(embdim, "zero", common.Float32)
	mstable := common.NewEmbeddingTable(embdim, "zero", common.Float32)

	for i := 0; i < size; i++ {
		rawGrad[i] = rand.Float32()
		rawParam[i] = rand.Float32()
		rawM[i] = rand.Float32()
		rawV[i] = rand.Float32()
		rawMaxSquare[i] = rand.Float32()
	}

	grad := common.NewTensor(rawGrad, dim)
	param := common.NewTensor(rawParam, dim)
	m := common.NewTensor(rawM, dim)
	v := common.NewTensor(rawV, dim)
	maxSquare := common.NewTensor(rawMaxSquare, dim)
	isgrad := common.NewIndexedSlices(grad, []int64{1})

	ptable.EmbeddingVectors[1] = param
	mtable.EmbeddingVectors[1] = m
	vtable.EmbeddingVectors[1] = v
	mstable.EmbeddingVectors[1] = maxSquare

	var lr float32 = 0.1
	var step int64 = 5
	var beta1 float32 = 0.9
	var beta2 float32 = 0.999
	var epsilon float32 = 1e-8

	expectedParam := make([]float32, size)
	expectedM := make([]float32, size)
	expectedV := make([]float32, size)
	expectedMaxSquare := make([]float32, size)

	for i := 0; i < size; i++ {
		expectedM[i] = beta1*rawM[i] + (1-beta1)*rawGrad[i]
		expectedV[i] = beta2*rawV[i] + (1-beta2)*rawGrad[i]*rawGrad[i]
		if rawMaxSquare[i] < expectedV[i] {
			expectedMaxSquare[i] = expectedV[i]
		} else {
			expectedMaxSquare[i] = rawMaxSquare[i]
		}

	}

	for i := 0; i < size; i++ {
		expectedParam[i] = rawParam[i] - lr*expectedM[i]/
			(1-float32(math.Pow(float64(beta1), float64(step))))/
			(float32(math.Sqrt(float64(expectedMaxSquare[i]/
				(1-float32(math.Pow(float64(beta2), float64(step)))))))+
				epsilon)
	}

	SparseAdam(isgrad, ptable, mtable, vtable, lr, step, beta1, beta2,
		epsilon, true, mstable)

	assert.True(t, common.CompareFloatArray(expectedM, common.Slice(m).([]float32), 0.00001))
	assert.True(t, common.CompareFloatArray(expectedV, common.Slice(v).([]float32), 0.00001))
	assert.True(t, common.CompareFloatArray(expectedParam, common.Slice(param).([]float32), 0.00001))
	assert.True(t, common.CompareFloatArray(expectedMaxSquare, common.Slice(maxSquare).([]float32), 0.00001))
}
