package kernelnew

import (
	"elasticdl.org/elasticdl/pkg/commonnew"
	"github.com/stretchr/testify/assert"
	"math"
	"math/rand"
	"testing"
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
	grad := commonnew.NewTensor(a, d)
	param := commonnew.NewTensor(b, d)

	expected := make([]float32, size)
	for i := 0; i < size; i++ {
		expected[i] = b[i] - lr*a[i]
	}

	err := SGD(grad, param, lr)
	assert.Nil(t, err)
	assert.Equal(t, expected, b)
}

func TestSparseSGD(t *testing.T) {
	a := []float32{-1.0, -1.0, -1.0, -1.0, -1.0, -1.0}
	d := []int64{3, 2}
	indices := []int64{1, 3, 3}
	grad := commonnew.NewTensor(a, d)
	isgrad := commonnew.NewIndexedSlices(grad, indices)

	table := commonnew.NewEmbeddingTable(2, "zero", commonnew.Float32)

	err := SparseSGD(isgrad, table, 0.1)
	assert.Nil(t, err)
	assert.Equal(t, 2, len(table.EmbeddingVectors))

	v1 := table.GetEmbeddingVector(1)
	assert.Equal(t, []float32{0.1, 0.1}, commonnew.Slice(v1).([]float32))

	v3 := table.GetEmbeddingVector(3)
	assert.Equal(t, []float32{0.2, 0.2}, commonnew.Slice(v3).([]float32))
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

	grad := commonnew.NewTensor(rawGrad, dim)
	param := commonnew.NewTensor(rawParam, dim)
	m := commonnew.NewTensor(rawM, dim)
	v := commonnew.NewTensor(rawV, dim)

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

	assert.True(t, commonnew.CompareFloatArray(expectedM, commonnew.Slice(m).([]float32), 0.0001))
	assert.True(t, commonnew.CompareFloatArray(expectedV, commonnew.Slice(v).([]float32), 0.00001))
	assert.True(t, commonnew.CompareFloatArray(expectedParam, commonnew.Slice(param).([]float32), 0.00001))
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

	grad := commonnew.NewTensor(rawGrad, dim)
	param := commonnew.NewTensor(rawParam, dim)
	m := commonnew.NewTensor(rawM, dim)
	v := commonnew.NewTensor(rawV, dim)
	maxSquare := commonnew.NewTensor(rawMaxSquare, dim)

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

	assert.True(t, commonnew.CompareFloatArray(expectedM, commonnew.Slice(m).([]float32), 0.00001))
	assert.True(t, commonnew.CompareFloatArray(expectedV, commonnew.Slice(v).([]float32), 0.00001))
	assert.True(t, commonnew.CompareFloatArray(expectedParam, commonnew.Slice(param).([]float32), 0.00001))
	assert.True(t, commonnew.CompareFloatArray(expectedMaxSquare, commonnew.Slice(maxSquare).([]float32), 0.00001))
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
	ptable := commonnew.NewEmbeddingTable(embdim, "zero", commonnew.Float32)
	mtable := commonnew.NewEmbeddingTable(embdim, "zero", commonnew.Float32)
	vtable := commonnew.NewEmbeddingTable(embdim, "zero", commonnew.Float32)
	mstable := commonnew.NewEmbeddingTable(embdim, "zero", commonnew.Float32)

	for i := 0; i < size; i++ {
		rawGrad[i] = rand.Float32()
		rawParam[i] = rand.Float32()
		rawM[i] = rand.Float32()
		rawV[i] = rand.Float32()
		rawMaxSquare[i] = rand.Float32()
	}

	grad := commonnew.NewTensor(rawGrad, dim)
	param := commonnew.NewTensor(rawParam, dim)
	m := commonnew.NewTensor(rawM, dim)
	v := commonnew.NewTensor(rawV, dim)
	maxSquare := commonnew.NewTensor(rawMaxSquare, dim)
	isgrad := commonnew.NewIndexedSlices(grad, []int64{1})

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

	assert.True(t, commonnew.CompareFloatArray(expectedM, commonnew.Slice(m).([]float32), 0.00001))
	assert.True(t, commonnew.CompareFloatArray(expectedV, commonnew.Slice(v).([]float32), 0.00001))
	assert.True(t, commonnew.CompareFloatArray(expectedParam, commonnew.Slice(param).([]float32), 0.00001))
	assert.True(t, commonnew.CompareFloatArray(expectedMaxSquare, commonnew.Slice(maxSquare).([]float32), 0.00001))
}
