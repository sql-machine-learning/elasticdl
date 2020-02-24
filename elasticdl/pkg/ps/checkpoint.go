package ps

import (
	"elasticdl.org/elasticdl/common"
	"elasticdl.org/elasticdl/proto"
	"fmt"
	go_pb "github.com/golang/protobuf/proto"
	"hash/fnv"
	"io/ioutil"
	"os"
)

func stringToID(name string, bucketNum int) int {
	h := fnv.New32a()
	h.Write([]byte(s))
	return h.Sum32() % bucketNum
}

func intToID(id int, bucketNum int) int {
	return id % bucketNum
}

func loadPBFromFile(string file) (*proto.Model, err) {
	b, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}
	res := &proto.Model{}
	go_pb.Unmarshal(b, res)
	return res, nil
}

func savePBToFile(pb *proto.Model, string file) {
	b, _ := go_pb.Marshal(pb)
	ioutil.WriteFile(file, b, os.ModePerm)
}

func loadModelShardFromPB(pb *proto.Model, shardID int, shardNum int) (map[string]*common.Tensor,
	map[string]*common.IndexedSlices) {
	denseParams := make(map[string]*common.Tensor)
	embeddingParams := make(map[string]*common.IndexedSlices)

	for name, v := range pb.DenseParameters {
		if stringToID(name) == shardID {
			denseParams[name] = common.DeserializeFromTensorProto(v)
		}
	}

	for name, v := range pb.EmbeddingTables {
		indexedSlices := common.DeserializeFromIndexedSliceProto(v)
		if indexedSlices != nil {
			ids := make([]int64)
			idsMap := make(map[int64]int64)
			for i, id := range indexedSlices.Ids {
				if intToID(id) == shardID {
					ids = append(ids, id)
					idsMap[id] = int64(i)
				}
			}
			height := int64(len(ids))
			width := indexedSlices.ConcatTensors.Dims[1]
			dtype := indexedSlices.ConcatTensors.Dtype
			tensor := common.NewEmptyTensor([]int64{height, width}, dtype)
			for i, id := range ids {
				tensor.SetRow(i, indexedSlices.ConcatTensors.GetRow(idsMap[id]))
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
		pb, err2 := loadPBFromFile(file)
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
			embeddingParams[k], err3 = common.MergeIndexedSlices(embeddingParams[k], v)
			if err3 != nil {
			    return nil err3
			}
		}
	}

	for k, v := range embeddingPairs {
		model.EmbeddingTables[k].SetEmbeddingVectors(v)
	}
	return model, nil
}
