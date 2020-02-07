package ps

import (
	"elasticdl.org/elasticdl/pkg/common"
	"elasticdl.org/elasticdl/pkg/proto"
	"fmt"
)

// Parameter contains non-embedding param and embedding param
type Parameter struct {
	NonEmbeddingParam map[string]*common.Tensor
	EmbeddingParam    map[string]*common.EmbeddingTable
	Version           int32
	InitStatus        bool
}

// NewParameter creates a parameter instance
func NewParameter() *Parameter {
	var p Parameter
	p.NonEmbeddingParam = make(map[string]*common.Tensor)
	p.EmbeddingParam = make(map[string]*common.EmbeddingTable)
	return &p
}

// GetNonEmbeddingParam returns non-embedding tensor pointer
func (p *Parameter) GetNonEmbeddingParam(name string) *common.Tensor {
	if value, ok := p.NonEmbeddingParam[name]; ok {
		return value
	}
	return nil
}

// GetEmbeddingParam returns embedding table pointer
func (p *Parameter) GetEmbeddingParam(name string) *common.EmbeddingTable {
	if value, ok := p.EmbeddingParam[name]; ok {
		return value
	}
	return nil
}

// SetEmbeddingParamInfo sets embedding table info of an embedding param
func (p *Parameter) SetEmbeddingParamInfo(name string, dim int64,
	initializer string) *common.EmbeddingTable {
	if _, ok := p.EmbeddingParam[name]; ok {
		return nil
	}
	t := common.NewEmbeddingTable(name, dim, initializer)
	p.EmbeddingParam[name] = t
	return t
}

// InitFromModelPB inits a Parameter instance from model PB to Parameter
func (p *Parameter) InitFromModelPB(pb *proto.Model) error {
	for _, v := range pb.EmbeddingTableInfo {
		p.SetEmbeddingParamInfo(v.Name, v.Dim, v.Initializer)
	}
	for _, v := range pb.Param {
		t := common.DeserializeTensorPB(v)
		if t.Indices == nil {
			p.NonEmbeddingParam[t.Name] = t
		} else {
			table := p.GetEmbeddingParam(t.Name)
			if table == nil {
				return fmt.Errorf("Embedding table %s is not created", t.Name)
			}
			err := p.EmbeddingParam[t.Name].SetEmbeddingVectors(t.Indices, t.Value)
			if err != nil {
				return err
			}
		}
	}
	if pb.Version >= 0 {
		p.Version = pb.Version
	}
	return nil
}
