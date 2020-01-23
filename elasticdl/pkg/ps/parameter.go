package ps

import "elasticdl.org/elasticdl/pkg/common"

// Parameter contains non-embedding param and embedding param
type Parameter struct {
	NonEmbeddingParam map[string]*common.Tensor
	EmbeddingParam    map[string]*common.EmbeddingTable
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
