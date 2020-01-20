package kernel

import "testing"
import "math/rand"
import "math"
import "github.com/stretchr/testify/assert"

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

	for i := 0; i < size; i++ {
		grad[i] = rand.Float32()
		param[i] = rand.Float32()
		m[i] = rand.Float32()
		v[i] = rand.Float32()
	}

	expected_param := make([]float32, size)
	expected_m := make([]float32, size)
	expected_v := make([]float32, size)

	for i := 0; i < size; i++ {
		expected_m[i] = float32(beta1) * m[i] + (1 - float32(beta1)) * grad[i]
		expected_v[i] = float32(beta2) * v[i] + (1 - float32(beta2)) * grad[i] * grad[i]
	}

	for i := 0; i < size; i++ {
		expected_param[i] = param[i] - float32(lr) * expected_m[i] / (1 - float32(math.Pow(beta1, float64(step)))) / (float32(math.Sqrt(float64(expected_v[i] / (1 - float32(math.Pow(beta2, float64(step)))))) + epsilon))
	}

	Adam(grad, param, m, v, lr, int64(size), step, beta1, beta2, epsilon)

	diff_param := 0
	diff_m := 0
	diff_v := 0

	for i := 0; i < size; i++ {
		if math.Abs(float64(expected_m[i]-m[i])) > 1e-6 {
			diff_m ++
		}
		if math.Abs(float64(expected_v[i]-v[i])) > 1e-6 {
			diff_v ++
		}
		if math.Abs(float64(expected_param[i]-param[i])) > 1e-6 {
			diff_param ++
		}
	}

	assert.Equal(t, diff_m, 0)
	assert.Equal(t, diff_v, 0)
	assert.Equal(t, diff_param, 0)
}