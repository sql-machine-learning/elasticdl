package ps

import "elasticdl.org/elasticdl/pkg/common"
import "elasticdl.org/elasticdl/pkg/kernel"
import "fmt"

// Optimizer interface
type Optimizer interface {
	GetLR() float32
	ApplyGradients([]common.Tensor, *Parameter) error
}

// SGDOptimizer struct
type SGDOptimizer struct {
	lr float32
}

// GetLR returns learning rate
func (opt *SGDOptimizer) GetLR() float32 {
	return opt.lr
}

// ApplyGradients applies gradients to parameters
func (opt *SGDOptimizer) ApplyGradients(grads []common.Tensor, p *Parameter) error {
	for _, grad := range grads {
		if grad.Indices == nil {
			t := p.GetNonEmbeddingParam(grad.Name)
			if t == nil {
				return fmt.Errorf("grad %s not in Parameter", grad.Name)
			}
			kernel.SGD(&grad, t, opt.GetLR())
		} else {
			t := p.GetEmbeddingParam(grad.Name)
			if t == nil {
				return fmt.Errorf("grad %s not in Parameter", grad.Name)
			}
			kernel.SparseSGD(&grad, t, opt.GetLR())
		}
	}
	return nil
}

// NewSGDOptimizer creates a SGD optimizer instance
func NewSGDOptimizer(lr float32) *SGDOptimizer {
	opt := SGDOptimizer{lr}
	return &opt
}
