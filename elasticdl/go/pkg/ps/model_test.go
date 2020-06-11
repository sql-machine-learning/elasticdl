// Copyright 2020 The SQLFlow Authors. All rights reserved.
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
)

func TestModelInit(t *testing.T) {
	d1 := []int64{2, 3}
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	t1 := common.NewTensor(v1, d1)

	d2 := []int64{2, 2}
	v2 := []float32{1.0, 2.0, 1.1, 2.2}
	t2 := common.NewTensor(v2, d2)

	model := NewModel()
	model.DenseParameters["t1"] = t1
	model.DenseParameters["t2"] = t2

	assert.Len(t, model.DenseParameters, 2)
	assert.Contains(t, model.DenseParameters, "t1")
	assert.Contains(t, model.DenseParameters, "t2")

	assert.Equal(t, model.GetDenseParameter("t1").Dims, d1)
	assert.Equal(t, model.GetDenseParameter("t2").Dims, d2)
	assert.Nil(t, model.GetDenseParameter("t3"))
}

func TestModelInitFrom(t *testing.T) {
	var modelPB = proto.Model{
		Version:             int32(1),
		EmbeddingTables:     make(map[string]*proto.IndexedSlicesProto),
		EmbeddingTableInfos: []*proto.EmbeddingTableInfo{},
	}
	d1 := []int64{3, 2}
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	t1 := common.NewTensor(v1, d1)

	i1 := []int64{1, 3, 5}
	var is = common.IndexedSlices{
		ConcatTensors: t1,
		Ids:           i1,
	}
	isPB := is.SerializeToIndexedSlicesProto()
	modelPB.EmbeddingTables["e1"] = isPB

	var epb = proto.EmbeddingTableInfo{
		Name:        "e1",
		Dim:         2,
		Initializer: "zero",
		Dtype:       common.Float32,
	}
	modelPB.EmbeddingTableInfos = append(modelPB.EmbeddingTableInfos, &epb)

	model := NewModel()
	assert.NotNil(t, model)
	err := model.InitFromModelPB(&modelPB)

	assert.Nil(t, err)
	assert.Contains(t, model.EmbeddingTables, "e1")

	e1 := model.GetEmbeddingTable("e1")
	assert.Equal(t, int64(2), e1.Dim)
	assert.Equal(t, 3, len(e1.EmbeddingVectors))

	ev1 := e1.GetEmbeddingVector(1)
	assert.True(t, common.CompareFloatArray([]float32{1.0, 2.0}, common.Slice(ev1).([]float32), 0.0001))

	ev3 := e1.GetEmbeddingVector(3)
	assert.True(t, common.CompareFloatArray([]float32{3.0, 4.0}, common.Slice(ev3).([]float32), 0.0001))

	ev5 := e1.GetEmbeddingVector(5)
	assert.True(t, common.CompareFloatArray([]float32{5.0, 6.0}, common.Slice(ev5).([]float32), 0.0001))
}

func TestModelSaveToPB(t *testing.T) {
	model := NewModel()

	d1 := []int64{3, 2}
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	t1 := common.NewTensor(v1, d1)

	i1 := []int64{1, 3, 5}
	is := common.NewIndexedSlices(t1, i1)

	model.EmbeddingTables["e1"] = common.NewEmbeddingTable(2, "zero", common.Float32)
	model.EmbeddingTables["e1"].SetEmbeddingVectors(is)

	modelPB := model.SaveToModelPB()
	assert.Equal(t, int64(2), modelPB.EmbeddingTableInfos[0].Dim)
	assert.Equal(t, "e1", modelPB.EmbeddingTableInfos[0].Name)

	resModel := NewModel()
	resModel.InitFromModelPB(modelPB)

	for _, i := range i1 {
		ev := model.EmbeddingTables["e1"].GetEmbeddingVector(i)
		rev := resModel.EmbeddingTables["e1"].GetEmbeddingVector(i)
		assert.True(t, common.CompareFloatArray(common.Slice(ev).([]float32),
			common.Slice(rev).([]float32), 0.0001))
	}
}
