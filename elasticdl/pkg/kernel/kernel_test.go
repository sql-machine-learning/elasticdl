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
	grad := common.Tensor{"t", a, d, nil}
	param := common.Tensor{"t", b, d, nil}

	expected := make([]float32, size)
	for i := 0; i < size; i++ {
		expected[i] = b[i] - lr*a[i]
	}

	err := SGD(&grad, &param, lr)
	assert.Nil(t, err)
	assert.Equal(t, expected, b)
}

func TestSparseSGD(t *testing.T) {
	a := []float32{-1.0, -1.0, -1.0, -1.0, -1.0, -1.0}
	d := []int64{3, 2}
	indices := []int64{1, 3, 3}
	grad := common.Tensor{"t", a, d, indices}

	table := common.NewEmbeddingTable("t", 2, "zero")

	err := SparseSGD(&grad, table, 0.1)
	assert.Nil(t, err)
	assert.Equal(t, 2, len(table.EmbeddingVector))

	v1 := table.GetEmbeddingVector(1)
	assert.Equal(t, 2, len(v1))
	assert.Equal(t, float32(0.1), v1[0])

	v3 := table.GetEmbeddingVector(3)
	assert.Equal(t, 2, len(v3))
	assert.Equal(t, float32(0.2), v3[0])
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

	grad := common.Tensor{"t", rawGrad, dim, nil}
	param := common.Tensor{"t", rawParam, dim, nil}
	m := common.Tensor{"t", rawM, dim, nil}
	v := common.Tensor{"t", rawV, dim, nil}

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
		expectedParam[i] = rawParam[i] - lr * expectedM[i]/
				(1-float32(math.Pow(float64(beta1), float64(step))))/
				(float32(math.Sqrt(float64(expectedV[i]/
					(1-float32(math.Pow(float64(beta2), float64(step)))))))+
					epsilon)
	}

	Adam(&grad, &param, &m, &v, lr, step, beta1, beta2,
		epsilon, false, nil)

	assert.True(t, common.CompareFloatArray(expectedM, m.Value, 0.0001))
	assert.True(t, common.CompareFloatArray(expectedV, v.Value, 0.00001))
	assert.True(t, common.CompareFloatArray(expectedParam, param.Value, 0.00001))
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

	grad := common.Tensor{"t", rawGrad, dim, nil}
	param := common.Tensor{"t", rawParam, dim, nil}
	m := common.Tensor{"t", rawM, dim, nil}
	v := common.Tensor{"t", rawV, dim, nil}
	maxSquare := common.Tensor{"t", rawMaxSquare, dim, nil}

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
		if rawMaxSquare[i] < expectedV[i]{
			expectedMaxSquare[i] = expectedV[i]
		} else {
			expectedMaxSquare[i] = rawMaxSquare[i]
		}
		
	}

	for i := 0; i < size; i++ {
		expectedParam[i] = rawParam[i] - lr * expectedM[i]/
				(1-float32(math.Pow(float64(beta1), float64(step))))/
				(float32(math.Sqrt(float64(expectedMaxSquare[i]/
					(1-float32(math.Pow(float64(beta2), float64(step)))))))+
					epsilon)
	}

	Adam(&grad, &param, &m, &v, lr, step, beta1, beta2,
		epsilon, true, &maxSquare)

	assert.True(t, common.CompareFloatArray(expectedM, m.Value, 0.00001))
	assert.True(t, common.CompareFloatArray(expectedV, v.Value, 0.00001))
	assert.True(t, common.CompareFloatArray(expectedParam, param.Value, 0.00001))
	assert.True(t, common.CompareFloatArray(expectedMaxSquare, maxSquare.Value, 0.00001))
}