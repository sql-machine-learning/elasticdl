// Copyright 2020 The ElasticDL Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ps

import (
	"testing"

	"elasticdl.org/elasticdl/pkg/common"
	"elasticdl.org/elasticdl/pkg/proto"
	"github.com/stretchr/testify/assert"
	"github.com/tensorflow/tensorflow/tensorflow/go/core/framework/tensor_go_proto"
)

func TestSGDOptimizer(t *testing.T) {
	d1 := []int64{2, 3}
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	t1 := common.NewTensor(v1, d1) //t1

	d2 := []int64{2, 2}
	v2 := []float32{1.0, 2.0, 1.1, 2.2}
	t2 := common.NewTensor(v2, d2) //t2

	model := NewModel()
	model.DenseParameters["t1"] = t1
	model.DenseParameters["t2"] = t2

	gv1 := []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
	gv2 := []float32{1.0, 1.0, 1.0, 1.0}
	grad1 := common.NewTensor(gv1, d1) //t1
	grad2 := common.NewTensor(gv2, d2) //t2
	pbModel := &proto.Model{
		DenseParameters: map[string]*tensor_go_proto.TensorProto{"t1": grad1.SerializeToTensorProto(), "t2": grad2.SerializeToTensorProto()},
	}

	opt := NewSGDOptimizer(0.1)

	// test dense parameter update
	err1 := opt.ApplyGradients(pbModel, model, float32(0.5)*opt.GetLR())
	assert.Equal(t, opt.GetLR(), float32(0.1))
	assert.Nil(t, err1)

	ev1 := []float32{0.95, 1.95, 2.95, 3.95, 4.95, 5.95}
	ev2 := []float32{0.95, 1.95, 1.05, 2.15}

	assert.True(t, common.CompareFloatArray(common.Slice(model.DenseParameters["t1"]).([]float32), ev1, 0.0001))
	assert.True(t, common.CompareFloatArray(common.Slice(model.DenseParameters["t2"]).([]float32), ev2, 0.0001))

	// test grad name error
	grad3 := common.NewTensor(gv2, d2) //t3
	pbModel = &proto.Model{
		DenseParameters: map[string]*tensor_go_proto.TensorProto{"t3": grad3.SerializeToTensorProto()},
	}
	err2 := opt.ApplyGradients(pbModel, model, float32(1.0)*opt.GetLR())
	assert.NotNil(t, err2)

	// test sparse parameter update
	info := &proto.EmbeddingTableInfo{
		Name:        "t3",
		Dim:         2,
		Initializer: "zero",
		Dtype:       common.Float32,
	}
	model.SetEmbeddingTableInfo(info)

	d3 := []int64{2, 2}
	v3 := []float32{1.0, 1.0, 1.0, 1.0}
	i3 := []int64{1, 3}
	grad3 = common.NewTensor(v3, d3) // t3
	sgrad3 := common.NewIndexedSlices(grad3, i3)

	pbModel = &proto.Model{
		DenseParameters: map[string]*tensor_go_proto.TensorProto{"t1": grad1.SerializeToTensorProto(), "t2": grad2.SerializeToTensorProto()},
		EmbeddingTables: map[string]*proto.IndexedSlicesProto{"t3": sgrad3.SerializeToIndexedSlicesProto()},
	}

	err3 := opt.ApplyGradients(pbModel, model, float32(1.0)*opt.GetLR())
	assert.Nil(t, err3)

	ev1 = []float32{0.85, 1.85, 2.85, 3.85, 4.85, 5.85}
	ev2 = []float32{0.85, 1.85, 0.95, 2.05}
	assert.True(t, common.CompareFloatArray(common.Slice(model.DenseParameters["t1"]).([]float32), ev1, 0.0001))
	assert.True(t, common.CompareFloatArray(common.Slice(model.DenseParameters["t2"]).([]float32), ev2, 0.0001))

	vectors := model.GetEmbeddingTable("t3").GetEmbeddingVectors(i3)
	expV := []float32{-0.1, -0.1, -0.1, -0.1}
	assert.True(t, common.CompareFloatArray(expV, common.Slice(vectors).([]float32), 0.0001))

	// more test for sparse parameter update
	d3 = []int64{4, 2}
	v3 = []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
	i3 = []int64{1, 3, 3, 5}
	grad3 = common.NewTensor(v3, d3) // t3
	sgrad3 = common.NewIndexedSlices(grad3, i3)

	pbModel = &proto.Model{
		EmbeddingTables: map[string]*proto.IndexedSlicesProto{"t3": sgrad3.SerializeToIndexedSlicesProto()},
	}

	err4 := opt.ApplyGradients(pbModel, model, float32(1.0)*opt.GetLR())
	assert.Nil(t, err4)

	vectors = model.GetEmbeddingTable("t3").GetEmbeddingVectors([]int64{1, 3, 5})
	expV = []float32{-0.2, -0.2, -0.3, -0.3, -0.1, -0.1}
	assert.True(t, common.CompareFloatArray(expV, common.Slice(vectors).([]float32), 0.0001))
}

func TestAdamOptimizer(t *testing.T) {
	d1 := []int64{2, 3}
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	t1 := common.NewTensor(v1, d1) //t1

	d2 := []int64{2, 2}
	v2 := []float32{1.0, 2.0, 1.1, 2.2}
	t2 := common.NewTensor(v2, d2) //t2

	model := NewModel()
	model.DenseParameters["t1"] = t1
	model.DenseParameters["t2"] = t2

	gv1 := []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
	gv2 := []float32{1.0, 1.0, 1.0, 1.0}
	grad1 := common.NewTensor(gv1, d1) //t1
	grad2 := common.NewTensor(gv2, d2) //t2
	pbModel := &proto.Model{
		DenseParameters: map[string]*tensor_go_proto.TensorProto{"t1": grad1.SerializeToTensorProto(), "t2": grad2.SerializeToTensorProto()},
		EmbeddingTableInfos: []*proto.EmbeddingTableInfo{
			&proto.EmbeddingTableInfo{
				Name:        "t3",
				Dim:         2,
				Initializer: "zero",
				Dtype:       common.Float32,
			},
		},
	}

	opt := NewAdamOptimizer(0.1, 0.9, 0.999, 1e-8, false)
	opt.InitOptimizer(pbModel)
	opt.step = 1

	// test dense parameter update
	err1 := opt.ApplyGradients(pbModel, model, float32(1.0)*opt.GetLR())
	assert.Equal(t, opt.GetLR(), float32(0.1))
	assert.Nil(t, err1)

	ev1 := []float32{0.9255863187, 1.9255863187, 2.9255863187, 3.9255863187, 4.9255863187, 5.9255863187}
	ev2 := []float32{0.9255863187, 1.9255863187, 1.0255863187, 2.1255863187}

	assert.True(t, common.CompareFloatArray(common.Slice(model.DenseParameters["t1"]).([]float32), ev1, 0.0001))
	assert.True(t, common.CompareFloatArray(common.Slice(model.DenseParameters["t2"]).([]float32), ev2, 0.0001))

	// test grad name error
	grad3 := common.NewTensor(gv2, d2) //t3
	pbModel = &proto.Model{
		DenseParameters: map[string]*tensor_go_proto.TensorProto{"t3": grad3.SerializeToTensorProto()},
	}
	err2 := opt.ApplyGradients(pbModel, model, float32(1.0)*opt.GetLR())
	assert.NotNil(t, err2)

	// test sparse parameter update
	info := &proto.EmbeddingTableInfo{
		Name:        "t3",
		Dim:         2,
		Initializer: "zero",
		Dtype:       common.Float32,
	}
	model.SetEmbeddingTableInfo(info)

	d3 := []int64{2, 2}
	v3 := []float32{1.0, 1.0, 1.0, 1.0}
	i3 := []int64{1, 3}
	grad3 = common.NewTensor(v3, d3) // t3
	sgrad3 := common.NewIndexedSlices(grad3, i3)

	pbModel = &proto.Model{
		DenseParameters: map[string]*tensor_go_proto.TensorProto{"t1": grad1.SerializeToTensorProto(), "t2": grad2.SerializeToTensorProto()},
		EmbeddingTables: map[string]*proto.IndexedSlicesProto{"t3": sgrad3.SerializeToIndexedSlicesProto()},
	}

	err3 := opt.ApplyGradients(pbModel, model, float32(1.0)*opt.GetLR())
	assert.Nil(t, err3)

	ev1 = []float32{0.8474920307, 1.8474920307, 2.8474920307, 3.8474920307, 4.8474920307, 5.8474920307}
	ev2 = []float32{0.8474920307, 1.8474920307, 0.9474920307, 2.0474920307}
	assert.True(t, common.CompareFloatArray(common.Slice(model.DenseParameters["t1"]).([]float32), ev1, 0.0001))
	assert.True(t, common.CompareFloatArray(common.Slice(model.DenseParameters["t2"]).([]float32), ev2, 0.0001))

	vectors := model.GetEmbeddingTable("t3").GetEmbeddingVectors(i3)
	expV := []float32{-0.058112835, -0.058112835, -0.058112835, -0.058112835}
	assert.True(t, common.CompareFloatArray(expV, common.Slice(vectors).([]float32), 0.0001))

	// more test for sparse parameter update
	d3 = []int64{3, 2}
	v3 = []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
	i3 = []int64{1, 3, 5}
	grad3 = common.NewTensor(v3, d3) // t3
	sgrad3 = common.NewIndexedSlices(grad3, i3)

	pbModel = &proto.Model{
		EmbeddingTables: map[string]*proto.IndexedSlicesProto{"t3": sgrad3.SerializeToIndexedSlicesProto()},
	}

	err4 := opt.ApplyGradients(pbModel, model, float32(1.0)*opt.GetLR())
	assert.Nil(t, err4)

	vectors = model.GetEmbeddingTable("t3").GetEmbeddingVectors([]int64{1, 3, 5})
	expV = []float32{-0.1314178004, -0.1314178004, -0.1314178004, -0.1314178004, -0.0545489238, -0.0545489238}
	assert.True(t, common.CompareFloatArray(expV, common.Slice(vectors).([]float32), 0.0001))
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

	optType = "Adagrad"
	optArgs = "learning_rate=0.2;epsilon=0.005;"
	opt, err = NewOptimizer(optType, optArgs)
	assert.Nil(t, err)
	adagradOpt, ok := opt.(*AdagradOptimizer)
	assert.True(t, ok)
	assert.Equal(t, adagradOpt.GetLR(), float32(0.2))
	assert.Equal(t, adagradOpt.epsilon, float32(0.005))
}
