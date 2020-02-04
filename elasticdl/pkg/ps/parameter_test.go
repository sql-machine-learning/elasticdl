package ps

import (
	"elasticdl.org/elasticdl/pkg/common"
	"elasticdl.org/elasticdl/pkg/proto"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestParameterInit(t *testing.T) {
	d1 := []int64{2, 3}
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	t1 := common.NewTensor("t1", v1, d1, nil)

	d2 := []int64{2, 2}
	v2 := []float32{1.0, 2.0, 1.1, 2.2}
	t2 := common.NewTensor("t2", v2, d2, nil)

	p := NewParameter(common.Float32Dtype)
	p.NonEmbeddingParam["t1"] = t1
	p.NonEmbeddingParam["t2"] = t2

	assert.Len(t, p.NonEmbeddingParam, 2)
	assert.Contains(t, p.NonEmbeddingParam, "t1")
	assert.Contains(t, p.NonEmbeddingParam, "t2")

	assert.Equal(t, []int64(p.GetNonEmbeddingParam("t1").Dim), d1)
	assert.Equal(t, []int64(p.GetNonEmbeddingParam("t2").Dim), d2)
	assert.Nil(t, p.GetNonEmbeddingParam("t3"))
}

func TestParameterDeserialize(t *testing.T) {
	var modelPB proto.Model
	modelPB.Version = int32(1)

	i1 := []int64{1, 3, 5}
	d1 := []int64{3, 2}
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	t1 := common.NewTensor("e1", v1, d1, i1)
	p1 := common.SerializeTensor(t1)
	modelPB.Param = append(modelPB.Param, p1)

	var epb proto.EmbeddingTableInfo
	epb.Name = "e1"
	epb.Dim = 2
	epb.Initializer = "zero"
	modelPB.EmbeddingTableInfo = append(modelPB.EmbeddingTableInfo, &epb)

	p, err := DeserializeModelPB(&modelPB)

	assert.Nil(t, err)
	assert.Contains(t, p.EmbeddingParam, "e1")

	e1 := p.GetEmbeddingParam("e1")
	assert.Equal(t, int64(2), e1.Dim)
	assert.Equal(t, 3, len(e1.EmbeddingVector))

	ev1 := e1.GetEmbeddingVector(1)
	assert.True(t, common.CompareFloatArray([]float32{1.0, 2.0}, ev1.InplaceSlice().([]float32), 0.0001))

	ev3 := e1.GetEmbeddingVector(3)
	assert.True(t, common.CompareFloatArray([]float32{3.0, 4.0}, ev3.InplaceSlice().([]float32), 0.0001))

	ev5 := e1.GetEmbeddingVector(5)
	assert.True(t, common.CompareFloatArray([]float32{5.0, 6.0}, ev5.InplaceSlice().([]float32), 0.0001))
}
