package kernel

import (
	"elasticdl.org/elasticdl/pkg/common"
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
	grad := common.NewTensor("t", a, d, nil)
	param := common.NewTensorInplace("t", b, d, nil)

	expected := make([]float32, size)
	for i := 0; i < size; i++ {
		expected[i] = b[i] - lr*a[i]
	}

	err := SGD(grad.Data, param.Data, lr)
	assert.Nil(t, err)
	assert.Equal(t, expected, b)
}

var a = []float32{-1.0, -1.0, -1.0, -1.0, -1.0, -1.0}
var d = []int64{3, 2}
var indices = []int64{1, 3, 3}
var grad = common.NewTensor("t", a, d, indices)
var table = common.NewEmbeddingTable("t", 2, "zero", common.Float32Dtype)

func TestSparseSGD(t *testing.T) {

	err := SparseSGD(grad, table, 0.1)
	assert.Nil(t, err)
	assert.Equal(t, 2, len(table.EmbeddingVector))

	v1 := table.GetEmbeddingVector(1)
	assert.Equal(t, 2, v1.Length)
	assert.True(t, common.CompareFloat(0.1, v1.At(0), 0.00001))

	v3 := table.GetEmbeddingVector(3)
	assert.Equal(t, 2, v3.Length)
	assert.True(t, common.CompareFloat(0.2, v3.At(0), 0.00001))
}

const size int = 10

var rawGrad = make([]float32, size)
var rawParam = make([]float32, size)
var rawM = make([]float32, size)
var rawV = make([]float32, size)
var rawMaxSquare = make([]float32, size)
var dim = []int64{2, 5}
var lr float32 = 0.1
var step int64 = 5
var beta1 float32 = 0.9
var beta2 float32 = 0.999
var epsilon float32 = 1e-8

func init() {
	for i := 0; i < size; i++ {
		rawGrad[i] = rand.Float32()
		rawParam[i] = rand.Float32()
		rawM[i] = rand.Float32()
		rawV[i] = rand.Float32()
		rawMaxSquare[i] = rand.Float32()
	}
}

func TestAdam(t *testing.T) {
	grad := common.NewTensor("t", rawGrad, dim, nil)
	param := common.NewTensor("t", rawParam, dim, nil)
	m := common.NewTensor("t", rawM, dim, nil)
	v := common.NewTensor("t", rawV, dim, nil)

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

	Adam(grad.Data, param.Data, m.Data, v.Data, lr, step, beta1, beta2,
		epsilon, false, nil)

	assert.True(t, common.CompareFloatArray(expectedM, m.InplaceSlice().([]float32), 0.0001))
	assert.True(t, common.CompareFloatArray(expectedV, v.InplaceSlice().([]float32), 0.00001))
	assert.True(t, common.CompareFloatArray(expectedParam, param.InplaceSlice().([]float32), 0.00001))
}

func TestAdamWithAmsgrad(t *testing.T) {
	grad := common.NewTensor("t", rawGrad, dim, nil)
	param := common.NewTensor("t", rawParam, dim, nil)
	m := common.NewTensor("t", rawM, dim, nil)
	v := common.NewTensor("t", rawV, dim, nil)
	maxSquare := common.NewTensor("t", rawMaxSquare, dim, nil)

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

	Adam(grad.Data, param.Data, m.Data, v.Data, lr, step, beta1, beta2,
		epsilon, true, maxSquare.Data)

	assert.True(t, common.CompareFloatArray(expectedM, m.InplaceSlice().([]float32), 0.00001))
	assert.True(t, common.CompareFloatArray(expectedV, v.InplaceSlice().([]float32), 0.00001))
	assert.True(t, common.CompareFloatArray(expectedParam, param.InplaceSlice().([]float32), 0.00001))
	assert.True(t, common.CompareFloatArray(expectedMaxSquare, maxSquare.InplaceSlice().([]float32), 0.00001))
}

func TestSparseAdam(t *testing.T) {
	ptable := common.NewEmbeddingTable("t", 10, "zero", common.Float32Dtype)
	mtable := common.NewEmbeddingTable("t", 10, "zero", common.Float32Dtype)
	vtable := common.NewEmbeddingTable("t", 10, "zero", common.Float32Dtype)
	mstable := common.NewEmbeddingTable("t", 10, "zero", common.Float32Dtype)

	grad := common.NewTensor("t", rawGrad, []int64{1, 10}, []int64{1})
	param := common.NewVector(rawParam)
	m := common.NewVector(rawM)
	v := common.NewVector(rawV)
	maxSquare := common.NewVector(rawMaxSquare)

	ptable.EmbeddingVector[1] = param
	mtable.EmbeddingVector[1] = m
	vtable.EmbeddingVector[1] = v
	mstable.EmbeddingVector[1] = maxSquare

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

	SparseAdam(grad, ptable, mtable, vtable, lr, step, beta1, beta2,
		epsilon, true, mstable)

	assert.True(t, common.CompareFloatArray(expectedM, m.InplaceSlice().([]float32), 0.00001))
	assert.True(t, common.CompareFloatArray(expectedV, v.InplaceSlice().([]float32), 0.00001))
	assert.True(t, common.CompareFloatArray(expectedParam, param.InplaceSlice().([]float32), 0.00001))
	assert.True(t, common.CompareFloatArray(expectedMaxSquare, maxSquare.InplaceSlice().([]float32), 0.00001))
}
