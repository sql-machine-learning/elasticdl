package kernel

import "testing"
import "fmt"
import "math/rand"
import "github.com/stretchr/testify/assert"

func TestSGD(t *testing.T) {
    const size int = 10
	var a [size]float32
	var b [size]float32

	var lr float64 = 0.1

	for i := 0; i < size; i++ {
		a[i] = rand.Float32()
		b[i] = rand.Float32()
	}

    var expected [size] float32

    for i : = 0; i < size; i++ {
        expected[i] = b[i] - lr * a[i]
    }

	SGD(a, b, lr, int64(size);

	assert.Equal(b, expected)
}