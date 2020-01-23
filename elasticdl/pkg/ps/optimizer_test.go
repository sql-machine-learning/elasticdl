package ps

import (
	"elasticdl.org/elasticdl/pkg/common"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSGDOptimizer(t *testing.T) {
	d1 := []int64{2, 3}
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	t1 := common.Tensor{"t1", v1, d1, nil}

	d2 := []int64{2, 2}
	v2 := []float32{1.0, 2.0, 1.1, 2.2}
	t2 := common.Tensor{"t2", v2, d2, nil}

	p := NewParameter()
	p.NonEmbeddingParam["t1"] = &t1
	p.NonEmbeddingParam["t2"] = &t2

	gv1 := []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
	gv2 := []float32{1.0, 1.0, 1.0, 1.0}
	grad1 := common.Tensor{"t1", gv1, d1, nil}
	grad2 := common.Tensor{"t2", gv2, d2, nil}
	grads := []common.Tensor{grad1, grad2}

	opt := NewSGDOptimizer(0.1)

	// test dense parameter update
	err1 := opt.ApplyGradients(grads, p)
	assert.Equal(t, opt.GetLR(), float32(0.1))
	assert.Nil(t, err1)

	ev1 := []float32{0.9, 1.9, 2.9, 3.9, 4.9, 5.9}
	ev2 := []float32{0.9, 1.9, 1.0, 2.1}

	assert.True(t, common.CompareFloatArray(p.NonEmbeddingParam["t1"].Value, ev1, 0.0001))
	assert.True(t, common.CompareFloatArray(p.NonEmbeddingParam["t2"].Value, ev2, 0.0001))

	// test grad name error
	grad3 := common.Tensor{"t3", gv2, d2, nil}
	grads = []common.Tensor{grad3}
	err2 := opt.ApplyGradients(grads, p)
	assert.NotNil(t, err2)

	// test sparse parameter update
	p.SetEmbeddingParamInfo("t3", 2, "zero")

	d3 := []int64{2, 2}
	v3 := []float32{1.0, 1.0, 1.0, 1.0}
	i3 := []int64{1, 3}
	grad3 = common.Tensor{"t3", v3, d3, i3}
	grads = []common.Tensor{grad1, grad2, grad3}

	err3 := opt.ApplyGradients(grads, p)
	assert.Nil(t, err3)

	ev1 = []float32{0.8, 1.8, 2.8, 3.8, 4.8, 5.8}
	ev2 = []float32{0.8, 1.8, 0.9, 2.0}
	assert.True(t, common.CompareFloatArray(p.NonEmbeddingParam["t1"].Value, ev1, 0.0001))
	assert.True(t, common.CompareFloatArray(p.NonEmbeddingParam["t2"].Value, ev2, 0.0001))

	vectors := p.GetEmbeddingParam("t3").GetEmbeddingVectors(i3)
	assert.Equal(t, 2, len(vectors))
	assert.True(t, common.CompareFloat(-0.1, vectors[0][0], 0.0001))
	assert.True(t, common.CompareFloat(-0.1, vectors[1][0], 0.0001))

	// more test for sparse parameter update
	d3 = []int64{4, 2}
	v3 = []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
	i3 = []int64{1, 3, 3, 5}
	grad3 = common.Tensor{"t3", v3, d3, i3}
	grads = []common.Tensor{grad3}

	err4 := opt.ApplyGradients(grads, p)
	assert.Nil(t, err4)

	vectors = p.GetEmbeddingParam("t3").GetEmbeddingVectors(i3)
	assert.Equal(t, 4, len(vectors))
	assert.True(t, common.CompareFloat(-0.2, vectors[0][0], 0.0001))
	assert.True(t, common.CompareFloat(-0.3, vectors[1][0], 0.0001))
	assert.True(t, common.CompareFloat(-0.1, vectors[3][0], 0.0001))
}
