package kernel

import "testing"
import "math/rand"
import "github.com/stretchr/testify/assert"
import "elasticdl.org/elasticdl/pkg/common"

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

	table := common.NewEmbeddingTable{"t", 2, "zero"}

	err := SparseSGD(&grad, &table, 0.1)
	assert.Nil(t, err)
	assert.Equal(t, 2, len(table.EmbeddingVector))

	v1 := (&table).GetEmbeddingVector(1)
	assert.Equal(t, 2, len(v1))
	assert.Equal(t, float32(0.1), v1[0])

	v3 := (&table).GetEmbeddingVector(3)
	assert.Equal(t, 2, len(v3))
	assert.Equal(t, float32(0.2), v3[0])
}
