package common

import "testing"
import "github.com/stretchr/testify/assert"

func TestTensorInit(t *testing.T) {
	i1 := []int64{1, 3, 5, 8, 9}
	d1 := []int64{2, 3}
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	t1 := Tensor{"t1", v1, d1, i1}

	assert.Equal(t, t1.name, "t1")
	assert.Equal(t, t1.value, v1)
	assert.Equal(t, t1.dim, d1)
	assert.Equal(t, t1.indices, i1)
}

func TestTensorSerialization(t *testing.T) {
}
