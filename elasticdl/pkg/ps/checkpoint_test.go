package ps

import (
	"elasticdl.org/elasticdl/common"
	"github.com/stretchr/testify/assert"
	"os"
	"path"
	"testing"
)

func TestCheckpoint(t *testing.T) {
	tmpDir := os.TempDir()
	tmpDir = path.Join(tmpDir, "TestCheckpoint")
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

	SaveModelToCheckpoint(tmpDir, model1, 0, bucketNum)
	SaveModelToCheckpoint(tmpDir, model2, 1, bucketNum)

	modelRes1, err1 := LoadModelFromCheckpoint(tmpDir, 0, 3)
	assert.Nil(t, err1)
	modelRes2, err2 := LoadModelFromCheckpoint(tmpDir, 1, 3)
	assert.Nil(t, err2)
	modelRes3, err3 := LoadModelFromCheckpoint(tmpDir, 2, 3)
	assert.Nil(t, err3)

	assert.Contains(t, modelRes1.EmbeddingTables["e1"].EmbeddingVectors, int64(0))
	assert.Contains(t, modelRes1.EmbeddingTables["e1"].EmbeddingVectors, int64(3))
	ev0 := model1.EmbeddingTables["e1"].GetEmbeddingVector(int64(0))
	rev0 := modelRes1.EmbeddingTables["e1"].GetEmbeddingVector(int64(0))
	assert.True(t, common.CompareFloatArray(common.Slice(ev0).([]float32),
		common.Slice(rev0).([]float32), 0.0001))

	assert.Contains(t, modelRes2.EmbeddingTables["e1"].EmbeddingVectors, int64(1))
	assert.Contains(t, modelRes2.EmbeddingTables["e1"].EmbeddingVectors, int64(4))
	ev1 := model2.EmbeddingTables["e1"].GetEmbeddingVector(int64(1))
	rev1 := modelRes2.EmbeddingTables["e1"].GetEmbeddingVector(int64(1))
	assert.True(t, common.CompareFloatArray(common.Slice(ev1).([]float32),
		common.Slice(rev1).([]float32), 0.0001))

	assert.Contains(t, modelRes3.EmbeddingTables["e1"].EmbeddingVectors, int64(2))
	assert.Contains(t, modelRes3.EmbeddingTables["e1"].EmbeddingVectors, int64(5))
	ev5 := model2.EmbeddingTables["e1"].GetEmbeddingVector(int64(5))
	rev5 := modelRes3.EmbeddingTables["e1"].GetEmbeddingVector(int64(5))
	assert.True(t, common.CompareFloatArray(common.Slice(ev5).([]float32),
		common.Slice(rev5).([]float32), 0.0001))

	os.RemoveAll(tmpDir)
}
