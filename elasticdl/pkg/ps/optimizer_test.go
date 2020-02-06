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
	grads := []*common.Tensor{&grad1, &grad2}

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
	grads = []*common.Tensor{&grad3}
	err2 := opt.ApplyGradients(grads, p)
	assert.NotNil(t, err2)

	// test sparse parameter update
	p.SetEmbeddingParamInfo("t3", 2, "zero")

	d3 := []int64{2, 2}
	v3 := []float32{1.0, 1.0, 1.0, 1.0}
	i3 := []int64{1, 3}
	grad3 = common.Tensor{"t3", v3, d3, i3}
	grads = []*common.Tensor{&grad1, &grad2, &grad3}

	err3 := opt.ApplyGradients(grads, p)
	assert.Nil(t, err3)

	ev1 = []float32{0.8, 1.8, 2.8, 3.8, 4.8, 5.8}
	ev2 = []float32{0.8, 1.8, 0.9, 2.0}
	assert.True(t, common.CompareFloatArray(p.NonEmbeddingParam["t1"].Value, ev1, 0.0001))
	assert.True(t, common.CompareFloatArray(p.NonEmbeddingParam["t2"].Value, ev2, 0.0001))

	vectors := p.GetEmbeddingParam("t3").GetEmbeddingVectors(i3)
	expV := []float32{-0.1, -0.1, -0.1, -0.1}
	assert.True(t, common.CompareFloatArray(expV, vectors.Value, 0.0001))

	// more test for sparse parameter update
	d3 = []int64{4, 2}
	v3 = []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
	i3 = []int64{1, 3, 3, 5}
	grad3 = common.Tensor{"t3", v3, d3, i3}
	grads = []*common.Tensor{&grad3}

	err4 := opt.ApplyGradients(grads, p)
	assert.Nil(t, err4)

	vectors = p.GetEmbeddingParam("t3").GetEmbeddingVectors([]int64{1, 3, 5})
	expV = []float32{-0.2, -0.2, -0.3, -0.3, -0.1, -0.1}
	assert.True(t, common.CompareFloatArray(expV, vectors.Value, 0.0001))
}

func TestAdamOptimizer(t *testing.T) {
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
	grads := []*common.Tensor{&grad1, &grad2}

	opt := NewAdamOptimizer(0.1, 0.9, 0.999, 1e-8, false)
	opt.step = 1

	opt.InitNonEmbeddingParam("t1", d1)
	opt.InitNonEmbeddingParam("t2", d2)

	// test dense parameter update
	err1 := opt.ApplyGradients(grads, p)
	assert.Equal(t, opt.BaseOptimizer.lr, float32(0.1))
	assert.Nil(t, err1)

	ev1 := []float32{0.9255863187, 1.9255863187, 2.9255863187, 3.9255863187, 4.9255863187, 5.9255863187}
	ev2 := []float32{0.9255863187, 1.9255863187, 1.0255863187, 2.1255863187}

	assert.True(t, common.CompareFloatArray(p.NonEmbeddingParam["t1"].Value, ev1, 0.0001))
	assert.True(t, common.CompareFloatArray(p.NonEmbeddingParam["t2"].Value, ev2, 0.0001))

	// test grad name error
	grad3 := common.Tensor{"t3", gv2, d2, nil}
	grads = []*common.Tensor{&grad3}
	err2 := opt.ApplyGradients(grads, p)
	assert.NotNil(t, err2)

	// test sparse parameter update
	p.SetEmbeddingParamInfo("t3", 2, "zero")
	opt.InitEmbeddingParam("t3", 2)

	d3 := []int64{2, 2}
	v3 := []float32{1.0, 1.0, 1.0, 1.0}
	i3 := []int64{1, 3}
	grad3 = common.Tensor{"t3", v3, d3, i3}
	grads = []*common.Tensor{&grad1, &grad2, &grad3}

	err3 := opt.ApplyGradients(grads, p)
	assert.Nil(t, err3)

	ev1 = []float32{0.8474920307, 1.8474920307, 2.8474920307, 3.8474920307, 4.8474920307, 5.8474920307}
	ev2 = []float32{0.8474920307, 1.8474920307, 0.9474920307, 2.0474920307}
	assert.True(t, common.CompareFloatArray(p.NonEmbeddingParam["t1"].Value, ev1, 0.0001))
	assert.True(t, common.CompareFloatArray(p.NonEmbeddingParam["t2"].Value, ev2, 0.0001))

	vectors := p.GetEmbeddingParam("t3").GetEmbeddingVectors(i3)
	expV := []float32{-0.058112835, -0.058112835, -0.058112835, -0.058112835}
	assert.True(t, common.CompareFloatArray(expV, vectors.Value, 0.0001))

	// more test for sparse parameter update
	d3 = []int64{4, 2}
	v3 = []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
	i3 = []int64{1, 3, 3, 5}
	grad3 = common.Tensor{"t3", v3, d3, i3}
	grads = []*common.Tensor{&grad3}

	err4 := opt.ApplyGradients(grads, p)
	assert.Nil(t, err4)

	vectors = p.GetEmbeddingParam("t3").GetEmbeddingVectors([]int64{1, 3, 5})
	expV = []float32{-0.1314178004, -0.1314178004, -0.2168087883, -0.2168087883, -0.0545489238, -0.0545489238}
	assert.True(t, common.CompareFloatArray(expV, vectors.Value, 0.0001))
}

func TestParseOptArgs(t *testing.T) {
	// parse SGD optimizer arguments
	optType := "SGD"
	optArgs := "learning_rate=0.1;momentum=0.0;nesterov=true;"
	argsMap, err := parseOptArgs(optType, optArgs)
	assert.Nil(t, err)
	assert.True(t, len(argsMap) == 3)
	assert.Equal(t, argsMap["learning_rate"], "0.1")
	assert.Equal(t, argsMap["momentum"], "0.0")
	assert.Equal(t, argsMap["nesterov"], "true")

	// should return error for redundant arguments
	optType = "SGD"
	optArgs = "learning_rate=0.1;momentum=0.0;nesterov=true;redundant_arg=1;"
	argsMap, err = parseOptArgs(optType, optArgs)
	assert.NotNil(t, err)

	// should return error for the learning rate
	optType = "SGD"
	optArgs = "momentum=0.0;nesterov=true;redundant_arg=1;"
	argsMap, err = parseOptArgs(optType, optArgs)
	assert.NotNil(t, err)

	// parse Adam optimizer arguments
	optType = "Adam"
	optArgs = "learning_rate=0.2;beta_1=0.5;beta_2=0.3;epsilon=0.005;amsgrad=false;"
	argsMap, err = parseOptArgs(optType, optArgs)
	assert.Nil(t, err)
	assert.True(t, len(argsMap) == 5)
	assert.Equal(t, argsMap["learning_rate"], "0.2")
	assert.Equal(t, argsMap["beta_1"], "0.5")
	assert.Equal(t, argsMap["beta_2"], "0.3")
	assert.Equal(t, argsMap["epsilon"], "0.005")
	assert.Equal(t, argsMap["amsgrad"], "false")
}

func TestNewOptimizer(t *testing.T) {
	optType := "SGD"
	optArgs := "learning_rate=0.1;momentum=0.0;nesterov=False;"
	opt, err := NewOptimizer(optType, optArgs)
	assert.Nil(t, err)
	assert.Equal(t, opt.GetLR(), float32(0.1))

	optType = "Adam"
	optArgs = "learning_rate=0.2;beta_1=0.5;beta_2=0.3;epsilon=0.005;amsgrad=false;"
	opt, err = NewOptimizer(optType, optArgs)
	assert.Nil(t, err)
	adamOpt, ok := opt.(*AdamOptimizer)
	assert.True(t, ok)
	assert.Equal(t, adamOpt.GetLR(), float32(0.2))
	assert.Equal(t, adamOpt.beta1, float32(0.5))
	assert.Equal(t, adamOpt.beta2, float32(0.3))
	assert.Equal(t, adamOpt.epsilon, float32(0.005))
	assert.False(t, adamOpt.amsgrad)
}
