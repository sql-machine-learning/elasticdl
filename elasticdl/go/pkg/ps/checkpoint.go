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
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/ioutil"
	"math/big"
	"os"
	"path"

	"elasticdl.org/elasticdl/pkg/common"
	"elasticdl.org/elasticdl/pkg/proto"
	go_pb "github.com/golang/protobuf/proto"
)

// StringToID maps a string to an id
func StringToID(name string, bucketNum int) int {
	input := []byte(name)
	sha256Bytes := sha256.Sum256(input)
	hexString := hex.EncodeToString(sha256Bytes[:])
	n := new(big.Int)
	res, _ := n.SetString(hexString, 32)
	bucketNumBig := new(big.Int).SetUint64(uint64(bucketNum))
	return int(res.Mod(res, bucketNumBig).Uint64())
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

// LoadModelFromCheckpoint loads model from checkpoint directory
func LoadModelFromCheckpoint(checkpointDir string, shardID int, shardNum int) (*Model, error) {
	files, err1 := ioutil.ReadDir(checkpointDir)
	if err1 != nil {
		return nil, err1
	}

	model := NewModel()
	embeddingParams := make(map[string]*common.IndexedSlices)
	for _, file := range files {
		pb, err2 := loadPBFromFile(path.Join(checkpointDir, file.Name()))
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

// SaveModelToCheckpoint saves in-memory model to checkpoint
func SaveModelToCheckpoint(checkpointDir string, model *Model, shardID int, shardNum int) {
	os.MkdirAll(checkpointDir, os.ModePerm)
	file := fmt.Sprintf("variables-%d-of-%d.ckpt", shardID, shardNum)
	modelPB := model.SaveToModelPB()
	savePBToFile(modelPB, path.Join(checkpointDir, file))
}
