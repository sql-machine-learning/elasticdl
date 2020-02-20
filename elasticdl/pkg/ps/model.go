package ps

import (
	"elasticdl.org/elasticdl/pkg/commonnew"
	"elasticdl.org/elasticdl/pkg/proto"
	"fmt"
)

// Model contains dense parameters and embedding tables
type Model struct {
	DenseParameters map[string]*commonnew.Tensor
	EmbeddingTables    map[string]*commonnew.EmbeddingTable
	Version           int32
	Initialized        bool
}

// NewModel creates a model instance
func NewModel() *Model {
    return &Model {
       DenseParameters: make(map[string]*commonnew.Tensor),
        EmbeddingTables : make(map[string]*commonnew.EmbeddingTable)
    }
}

// GetDenseParameter returns dense parameter pointer
func (p *Model) GetDenseParameter(name string) *commonnew.Tensor {
	if value, ok := p.DenseParameters[name]; ok {
		return value
	}
	return nil
}

// GetEmbeddingTable returns embedding table pointer
func (p *Model) GetEmbeddingTable(name string) *commonnew.EmbeddingTable {
	if value, ok := p.EmbeddingTables[name]; ok {
		return value
	}
	return nil
}

// SetEmbeddingTableInfo sets embedding table info of an embedding param
func (model *Model) SetEmbeddingTableInfo(info *proto.EmbeddingTableInfo) {
	if _, ok := model.EmbeddingTables[info.Name]; ok {
		return
	}
	t := common.NewEmbeddingTable(info.Dim, info.Initializer, info.Dtype)
	model.EmbeddingTables[info.Name] = t
}

// InitFromModelPB inits a Parameter instance from model PB to Parameter
func (p *Parameter) InitFromModelPB(pb *proto.Model) error {
	for _, v := range pb.EmbeddingTableInfo {
		model.SetEmbeddingTableInfo(v)
	}
	for name, v := range pb.DenseParameters {
		model.DenseParameters[name] = DeserializeFromTensorPB(v)
	}
	for name, v := range pb.EmbeddingTables {
	    table := p.GetEmbeddingTable(name)
	    if table == nil {
			return fmt.Errorf("Embedding table %s is not created", name)
		}
        err := model.EmbeddingTables[name].SetEmbeddingVectors(v)
        if err != nil {
				return err
			}
	}
	if pb.Version >= 0 {
		p.Version = pb.Version
	}
	return nil
}
