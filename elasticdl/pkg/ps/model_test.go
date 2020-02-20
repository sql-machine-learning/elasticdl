package ps

import (
	"elasticdl.org/elasticdl/pkg/commonnew"
	"elasticdl.org/elasticdl/pkg/proto"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestModelInit(t *testing.T) {
	d1 := []int64{2, 3}
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	t1 := commonnew.NewTensor(v1, d1)

	d2 := []int64{2, 2}
	v2 := []float32{1.0, 2.0, 1.1, 2.2}
	t2 := commonnew.NewTensor(v2, d2)

	model := NewModel()
	model.DenseParameters["t1"] = &t1
	model.DenseParameters["t2"] = &t2

	assert.Len(t, p.DenseParameters, 2)
	assert.Contains(t, p.DenseParameters, "t1")
	assert.Contains(t, p.DenseParameters, "t2")

	assert.Equal(t, p.DenseParameters("t1").Dims, d1)
	assert.Equal(t, p.DenseParameters("t2").Dims, d2)
	assert.Nil(t, p.DenseParameters("t3"))
}

func TestModelInitFrom(t *testing.T) {
	var modelPB proto.Model
	modelPB.Version = int32(1)
	d1 := []int64{3, 2}
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	t1 := commonnew.NewTensor(v1, d1)

	i1 := []int64{1, 3, 5}
	p1 := commonnew.SerializeToIndexedSlices(i1)
	modelPB.EmbeddingTables["e1"] = p1

	var epb = proto.EmbeddingTableInfo{
		Name:        "e1",
		Dim:         2,
		Initializer: "zero",
		Dtype:       commonnew.Float32,
	}
	modelPB.EmbeddingTableInfo = append(modelPB.EmbeddingTableInfo, &epb)

	model := NewModel()
	err := model.InitFromModelPB(&modelPB)

	assert.Nil(t, err)
	assert.Contains(t, p.EmbeddingTables, "e1")

	e1 := p.GetEmbeddingTable("e1")
	assert.Equal(t, int64(2), e1.Dim)
	assert.Equal(t, 3, len(e1.EmbeddingVectors))

	ev1 := e1.GetEmbeddingVector(1)
	assert.True(t, commonnew.CompareFloatArray([]float32{1.0, 2.0}, commonnew.Slice(ev1).([]float64), 0.0001))

	ev3 := e1.GetEmbeddingVector(3)
	assert.True(t, common.CompareFloatArray([]float32{3.0, 4.0}, commonnew.Slice(ev3).([]float64), 0.0001))

	ev5 := e1.GetEmbeddingVector(5)
	assert.True(t, common.CompareFloatArray([]float32{5.0, 6.0}, commonnew.Slice(ev5).([]float64), 0.0001))
}
