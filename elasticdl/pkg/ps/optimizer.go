package ps

import "elasticdl.org/elasticdl/pkg/common"
import "elasticdl.org/elasticdl/pkg/kernel"
import "fmt"

// Optimizer interface
type Optimizer interface {
	GetLR() float64
	ApplyGradients([]common.Tensor, *Parameter) error
}

// SGDOptimizer struct
type SGDOptimizer struct {
	lr float64
}

// GetLR returns learning rate
func (opt *SGDOptimizer) GetLR() float64 {
	return opt.lr
}

// ApplyGradients applies gradients to parameters
func (opt *SGDOptimizer) ApplyGradients(grads []common.Tensor, p *Parameter) error {
	for _, grad := range grads {
		t := p.GetNonEmbeddingParam(grad.Name)
		if t == nil {
			return fmt.Errorf("grad %s not in Parameter", grad.Name)
		}
		kernel.SGD(grad.Value, t.Value, opt.GetLR(), int64(len(grad.Value)))
	}
	return nil
}
