package ps

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestParameterInit(t *testing.T) {
	d1 := []int64{2, 3}
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	t1 := Tensor{"t1", v1, d1}

	d2 := []int64{2, 2}
	v2 := []float32{1.0, 2.0, 1.1, 2.2}
	t2 := Tensor{"t2", v2, d2}

	var p Parameter
	p.NonEmbeddingParam = make(map[string]common.Tensor)

	p.NonEmbeddingParam["t1"] = t1
	p.NonEmbeddingParam["t2"] = t2

	assert.Len(t, p.NonEmbeddingParam, 2)
	assert.Contains(t, p.NonEmbeddingParam, "t1")
	assert.Contains(t, p.NonEmbeddingParam, "t2")
	assert.Equal(t, p.NonEmbeddingParam["t1"].Dim, d1)
	assert.Equal(t, p.NonEmbeddingParam["t2"].Dim, d2)
}
