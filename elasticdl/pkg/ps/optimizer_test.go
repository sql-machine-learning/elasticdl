package ps

import (
	"elasticdl.org/elasticdl/pkg/common"
	"github.com/stretchr/testify/assert"
	"testing"
)

var d1 = []int64{2, 3}
var v1 = []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
var d2 = []int64{2, 2}
var v2 = []float64{1.0, 2.0, 1.1, 2.2}
var gv1 = []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
var gv2 = []float64{1.0, 1.0, 1.0, 1.0}
var d3 = []int64{2, 2}
var v3 = []float64{1.0, 1.0, 1.0, 1.0}
var i3 = []int64{1, 3}
var d4 = []int64{4, 2}
var v4 = []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
var i4 = []int64{1, 3, 3, 5}

var grad1 = common.NewTensor("t1", gv1, d1, nil)
var grad2 = common.NewTensor("t2", gv2, d2, nil)
var grad3 = common.NewTensor("t3", gv2, d2, nil)
var grad4 = common.NewTensor("t3", v3, d3, i3)
var grad5 = common.NewTensor("t3", v4, d4, i4)

func TestSGDOptimizer(t *testing.T) {
	t1 := common.NewTensor("t1", v1, d1, nil)
	t2 := common.NewTensor("t2", v2, d2, nil)

	p := NewParameter(common.Float64)
	p.NonEmbeddingParam["t1"] = t1
	p.NonEmbeddingParam["t2"] = t2

	grads := []*common.Tensor{grad1, grad2}

	opt := NewSGDOptimizer(0.1)

	// test dense parameter update
	err1 := opt.ApplyGradients(grads, p)
	assert.True(t, common.CompareFloat(float64(opt.GetLR()), float64(0.1), 0.00001))
	assert.Nil(t, err1)

	ev1 := []float64{0.9, 1.9, 2.9, 3.9, 4.9, 5.9}
	ev2 := []float64{0.9, 1.9, 1.0, 2.1}

	assert.True(t, common.CompareFloatArray(p.NonEmbeddingParam["t1"].InplaceSlice().([]float64), ev1, 0.0001))
	assert.True(t, common.CompareFloatArray(p.NonEmbeddingParam["t2"].InplaceSlice().([]float64), ev2, 0.0001))

	// test grad name error
	grads = []*common.Tensor{grad3}
	err2 := opt.ApplyGradients(grads, p)
	assert.NotNil(t, err2)

	// test sparse parameter update
	p.SetEmbeddingParamInfo("t3", 2, "zero")

	grads = []*common.Tensor{grad1, grad2, grad4}

	err3 := opt.ApplyGradients(grads, p)
	assert.Nil(t, err3)

	ev1 = []float64{0.8, 1.8, 2.8, 3.8, 4.8, 5.8}
	ev2 = []float64{0.8, 1.8, 0.9, 2.0}
	assert.True(t, common.CompareFloatArray(p.NonEmbeddingParam["t1"].InplaceSlice().([]float64), ev1, 0.0001))
	assert.True(t, common.CompareFloatArray(p.NonEmbeddingParam["t2"].InplaceSlice().([]float64), ev2, 0.0001))

	vectors := p.GetEmbeddingParam("t3").GetEmbeddingVectors(i3)
	expV := []float64{-0.1, -0.1, -0.1, -0.1}
	assert.True(t, common.CompareFloatArray(expV, vectors.InplaceSlice().([]float64), 0.0001))

	// more test for sparse parameter update
	grads = []*common.Tensor{grad5}

	err4 := opt.ApplyGradients(grads, p)
	assert.Nil(t, err4)

	vectors = p.GetEmbeddingParam("t3").GetEmbeddingVectors([]int64{1, 3, 5})
	expV = []float64{-0.2, -0.2, -0.3, -0.3, -0.1, -0.1}
	assert.True(t, common.CompareFloatArray(expV, vectors.InplaceSlice().([]float64), 0.0001))
}

func TestAdamOptimizer(t *testing.T) {
	t1 := common.NewTensor("t1", v1, d1, nil)
	t2 := common.NewTensor("t2", v2, d2, nil)

	p := NewParameter(common.Float64)
	p.NonEmbeddingParam["t1"] = t1
	p.NonEmbeddingParam["t2"] = t2

	grads := []*common.Tensor{grad1, grad2}

	opt := NewAdamOptimizer(0.1, 0.9, 0.999, 1e-8, false, common.Float64)
	opt.step = 1

	opt.InitNonEmbeddingParam("t1", d1)
	opt.InitNonEmbeddingParam("t2", d2)

	// test dense parameter update
	err1 := opt.ApplyGradients(grads, p)
	assert.True(t, common.CompareFloat(float64(opt.GetLR()), float64(0.1), 0.00001))
	assert.Nil(t, err1)

	ev1 := []float64{0.9255863187, 1.9255863187, 2.9255863187, 3.9255863187, 4.9255863187, 5.9255863187}
	ev2 := []float64{0.9255863187, 1.9255863187, 1.0255863187, 2.1255863187}

	assert.True(t, common.CompareFloatArray(p.NonEmbeddingParam["t1"].InplaceSlice().([]float64), ev1, 0.0001))
	assert.True(t, common.CompareFloatArray(p.NonEmbeddingParam["t2"].InplaceSlice().([]float64), ev2, 0.0001))

	// test grad name error
	grads = []*common.Tensor{grad3}
	err2 := opt.ApplyGradients(grads, p)
	assert.NotNil(t, err2)

	// test sparse parameter update
	p.SetEmbeddingParamInfo("t3", 2, "zero")
	opt.InitEmbeddingParam("t3", 2)

	grads = []*common.Tensor{grad1, grad2, grad4}

	err3 := opt.ApplyGradients(grads, p)
	assert.Nil(t, err3)

	ev1 = []float64{0.8474920307, 1.8474920307, 2.8474920307, 3.8474920307, 4.8474920307, 5.8474920307}
	ev2 = []float64{0.8474920307, 1.8474920307, 0.9474920307, 2.0474920307}
	assert.True(t, common.CompareFloatArray(p.NonEmbeddingParam["t1"].InplaceSlice().([]float64), ev1, 0.0001))
	assert.True(t, common.CompareFloatArray(p.NonEmbeddingParam["t2"].InplaceSlice().([]float64), ev2, 0.0001))

	vectors := p.GetEmbeddingParam("t3").GetEmbeddingVectors(i3)
	expV := []float64{-0.058112835, -0.058112835, -0.058112835, -0.058112835}
	assert.True(t, common.CompareFloatArray(expV, vectors.InplaceSlice().([]float64), 0.0001))

	// more test for sparse parameter update
	grads = []*common.Tensor{grad5}

	err4 := opt.ApplyGradients(grads, p)
	assert.Nil(t, err4)

	vectors = p.GetEmbeddingParam("t3").GetEmbeddingVectors([]int64{1, 3, 5})
	expV = []float64{-0.1314178004, -0.1314178004, -0.2168087883, -0.2168087883, -0.0545489238, -0.0545489238}
	assert.True(t, common.CompareFloatArray(expV, vectors.InplaceSlice().([]float64), 0.0001))
}
