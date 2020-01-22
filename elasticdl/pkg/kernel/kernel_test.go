package kernel

import "testing"
import "math/rand"
import "math"
import "github.com/stretchr/testify/assert"
import "github.com/google/go-cmp/cmp"

func TestSGD(t *testing.T) {
	const size int = 10
	a := make([]float32, size)
	b := make([]float32, size)
	var lr float32 = 0.1

	for i := 0; i < size; i++ {
		a[i] = rand.Float32()
		b[i] = rand.Float32()
	}

	expected := make([]float32, size)

	for i := 0; i < size; i++ {
		expected[i] = b[i] - lr*a[i]
	}

	SGD(a, b, float64(lr), int64(size))

	assert.Equal(t, b, expected)
}

func TestAdam(t *testing.T) {
	const size int = 10
	grad := make([]float32, size)
	param := make([]float32, size)
	m := make([]float32, size)
	v := make([]float32, size)
	var lr float64 = 0.1
	var step int64 = 5
	var beta1 float64 = 0.9
	var beta2 float64 = 0.999
	var epsilon float64 = 1e-8

	const tolerance = .00001
	opt := cmp.Comparer(func(x, y float32) bool {
		diff := math.Abs(float64(x - y))
		mean := math.Abs(float64(x+y)) / 2.0
		if math.IsNaN(diff / mean) {
			return true
		}
		return (diff / mean) < tolerance
	})

	for i := 0; i < size; i++ {
		grad[i] = rand.Float32()
		param[i] = rand.Float32()
		m[i] = rand.Float32()
		v[i] = rand.Float32()
	}

	expectedParam := make([]float32, size)
	expectedM := make([]float32, size)
	expectedV := make([]float32, size)

	for i := 0; i < size; i++ {
		expectedM[i] = float32(beta1)*m[i] + (1-float32(beta1))*grad[i]
		expectedV[i] = float32(beta2)*v[i] + (1-float32(beta2))*grad[i]*grad[i]
	}

	for i := 0; i < size; i++ {
		expectedParam[i] = param[i] - float32(lr)*expectedM[i]/(1-float32(math.Pow(beta1, float64(step))))/(float32(math.Sqrt(float64(expectedV[i]/(1-float32(math.Pow(beta2, float64(step))))))+epsilon))
	}

	Adam(grad, param, m, v, lr, int64(size), step, beta1, beta2, epsilon, false, nil)

	diffParam := 0
	diffM := 0
	diffV := 0

	for i := 0; i < size; i++ {
		if !cmp.Equal(expectedM[i], m[i], opt) {
			diffM++
		}
		if !cmp.Equal(expectedV[i], v[i], opt) {
			diffV++
		}
		if !cmp.Equal(expectedParam[i], param[i], opt) {
			diffParam++
		}
	}

	assert.Equal(t, diffM, 0)
	assert.Equal(t, diffV, 0)
	assert.Equal(t, diffParam, 0)
}
