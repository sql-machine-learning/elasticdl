package ps

import (
	"elasticdl.org/elasticdl/common"
	"fmt"
	"github.com/stretchr/testify/assert"
	"os"
	"testing"
)

func TestCheckPoint(t *testing.T) {
	tmpDir := os.TempDir()
	fmt.Println(tmpDir)
	var bucketNum int = 2

	model1 := NewModel()
	d1 := []int64{3, 2}
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	t1 := common.NewTensor(v1, d1)
	i1 := []int64{0, 2, 4}
	is1 := common.NewIndexedSlices(t1, i1)

	model1.EmbeddingTables["e1"] = common.NewEmbeddingTable(2, "zero", common.Float32)
	model1.EmbeddingTables["e1"].SetEmbeddingVectors(is1)

	model2 := NewModel()
	d2 := []int64{3, 2}
	v2 := []float32{6.0, 5.0, 4.0, 3.0, 2.0, 1.0}
	t2 := common.NewTensor(v2, d2)
	i2 := []int64{1, 3, 5}
	is2 := common.NewIndexedSlices(t2, i2)

	model2.EmbeddingTables["e1"] = common.NewEmbeddingTable(2, "zero", common.Float32)
	model2.EmbeddingTables["e1"].SetEmbeddingVectors(is2)

	SaveModelToCheckPoint(tmpDir, model1, 0, bucketNum)
	SaveModelToCheckPoint(tmpDir, model2, 1, bucketNum)

	modelRes1, err1 := LoadModelFromCheckPoint(tmpDir, 0, bucketNum)
	assert.Nil(t, err1)
	modelRes2, err2 := LoadModelFromCheckPoint(tmpDir, 1, bucketNum)
	assert.Nil(t, err2)

	for _, i := range i1 {
		ev := model1.EmbeddingTables["e1"].GetEmbeddingVector(i)
		rev := modelRes1.EmbeddingTables["e1"].GetEmbeddingVector(i)
		assert.True(t, common.CompareFloatArray(common.Slice(ev).([]float32),
			common.Slice(rev).([]float32), 0.0001))
	}

	for _, i := range i2 {
		ev := model2.EmbeddingTables["e1"].GetEmbeddingVector(i)
		rev := modelRes2.EmbeddingTables["e1"].GetEmbeddingVector(i)
		assert.True(t, common.CompareFloatArray(common.Slice(ev).([]float32),
			common.Slice(rev).([]float32), 0.0001))
	}
}
