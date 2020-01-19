package ps

import "elasticdl.org/elasticdl/pkg/common"

// Parameter contains non-embedding param and embedding param
type Parameter struct {
	NonEmbeddingParam map[string]common.Tensor
}
