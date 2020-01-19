package ps

import "elasticdl.org/elasticdl/common/tensor"

// Parameter contains non-embedding param and embedding param
type Parameter struct {
	NonEmbeddingParam map[string]common.Tensor
}
