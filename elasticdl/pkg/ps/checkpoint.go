package ps

import (
	"elasticdl.org/elasticdl/common"
	"elasticdl.org/elasticdl/proto"
	"fmt"
	go_pb "github.com/golang/protobuf/proto"
	"hash/fnv"
	"io/ioutil"
	"os"
	"path"
)

// StringToID maps a string to an id
func StringToID(name string, bucketNum int) int {
	h := fnv.New32a()
	h.Write([]byte(name))
	return int(h.Sum32()) % bucketNum
}

// IntToID maps an int to an id
func IntToID(id int64, bucketNum int) int {
	return int(id % int64(bucketNum))
}

func loadPBFromFile(file string) (*proto.Model, error) {
	b, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}
	res := &proto.Model{}
	go_pb.Unmarshal(b, res)
	return res, nil
}

func savePBToFile(pb *proto.Model, file string) {
	b, _ := go_pb.Marshal(pb)
	ioutil.WriteFile(file, b, os.ModePerm)
}

func loadModelShardFromPB(pb *proto.Model, shardID int, shardNum int) (map[string]*common.Tensor,
	map[string]*common.IndexedSlices) {
	denseParams := make(map[string]*common.Tensor)
	embeddingParams := make(map[string]*common.IndexedSlices)

	for name, v := range pb.DenseParameters {
		if StringToID(name, shardNum) == shardID {
			denseParams[name] = common.DeserializeFromTensorProto(v)
		}
	}

	for name, v := range pb.EmbeddingTables {
		indexedSlices := common.DeserializeFromIndexedSliceProto(v)
		if indexedSlices != nil {
			var ids []int64
			idsMap := make(map[int64]int64)
			for i, id := range indexedSlices.Ids {
				if IntToID(id, shardNum) == shardID {
					ids = append(ids, id)
					idsMap[id] = int64(i)
				}
			}
			height := int64(len(ids))
			width := indexedSlices.ConcatTensors.Dims[1]
			dtype := indexedSlices.ConcatTensors.Dtype
			tensor := common.NewEmptyTensor([]int64{height, width}, dtype)
			for i, id := range ids {
				tensor.SetRow(int64(i), indexedSlices.ConcatTensors.GetRow(idsMap[id]))
			}
			is := common.NewIndexedSlices(tensor, ids)
			embeddingParams[name] = is
		}
	}
	return denseParams, embeddingParams
}

// LoadModelFromCheckPoint loads model from checkpoint directory
func LoadModelFromCheckPoint(checkPointDir string, shardID int, shardNum int) (*Model, error) {
	files, err1 := ioutil.ReadDir(checkPointDir)
	if err1 != nil {
		return nil, err1
	}

	model := NewModel()
	embeddingParams := make(map[string]*common.IndexedSlices)
	for _, file := range files {
		pb, err2 := loadPBFromFile(path.Join(checkPointDir, file.Name()))
		if err2 != nil {
			return nil, err2
		}

		for _, info := range pb.EmbeddingTableInfos {
			model.SetEmbeddingTableInfo(info)
		}

		dp, ep := loadModelShardFromPB(pb, shardID, shardNum)
		for k, v := range dp {
			model.DenseParameters[k] = v
		}
		for k, v := range ep {
			var err3 error
			embeddingParams[k], err3 = common.MergeIndexedSlices(embeddingParams[k], v)
			if err3 != nil {
				return nil, err3
			}
		}
	}

	for k, v := range embeddingParams {
		model.EmbeddingTables[k].SetEmbeddingVectors(v)
	}
	return model, nil
}

// SaveModelToCheckPoint saves in-memory model to checkpoint
func SaveModelToCheckPoint(checkPointDir string, model *Model, shardID int, shardNum int) {
	os.MkdirAll(checkPointDir, os.ModePerm)
	file := fmt.Sprintf("variables-%d-of-%d.ckpt", shardID, shardNum)
	modelPB := model.SaveToModelPB()
	savePBToFile(modelPB, path.Join(checkPointDir, file))
}
