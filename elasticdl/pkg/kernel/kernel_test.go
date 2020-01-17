package kernel

import "testing"
import "math/rand"
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
