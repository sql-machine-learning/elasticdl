package ps

import "elasticdl.org/elasticdl/pkg/common"

// Parameter contains non-embedding param and embedding param
type Parameter struct {
	NonEmbeddingParam map[string]common.Tensor
}

// GetNonEmbeddingParam returns non-embedding tensor pointer
func (p *Parameter) GetNonEmbeddingParam(name string) *common.Tensor {
	if value, ok := p.NonEmbeddingParam[name]; ok {
		return &value
	}
	return nil
}
