package ps

import "elasticdl.org/elasticdl/pkg/common"
import "elasticdl.org/elasticdl/pkg/kernel"
import "fmt"

// Optimizer interface
type Optimizer interface {
	GetLR() float32
	ApplyGradients([]common.Tensor, *Parameter) error
}

// BaseOptimizer struct
type BaseOptimizer struct {
	lr   float32
	step int64
}

// SGDOptimizer struct
type SGDOptimizer struct {
	BaseOptimizer
}

// GetLR returns learning rate SGD
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
	var opt SGDOptimizer
	opt.lr = lr
	return &opt
}

// AdamOptimizer struct
type AdamOptimizer struct {
	BaseOptimizer
	beta1     float32
	beta2     float32
	epsilon   float32
	amsgrad   bool
	m         *Parameter
	v         *Parameter
	maxSquare *Parameter
}

// ApplyGradients applies gradients to parameters
func (opt *AdamOptimizer) ApplyGradients(grads []common.Tensor, p *Parameter) error {
	opt.step++
	for _, grad := range grads {
		if grad.Indices == nil {
			t := p.GetNonEmbeddingParam(grad.Name)
			m := opt.m.GetNonEmbeddingParam(grad.Name)
			v := opt.v.GetNonEmbeddingParam(grad.Name)
			if t == nil || m == nil || v == nil {
				return fmt.Errorf("grad %s not in Parameter", grad.Name)
			}
			if opt.amsgrad {
				ms := opt.maxSquare.GetNonEmbeddingParam(grad.Name)
				kernel.Adam(&grad, t, m, v, opt.lr, opt.step, opt.beta1, opt.beta2, opt.epsilon, true, ms)
			} else {
				kernel.Adam(&grad, t, m, v, opt.lr, opt.step, opt.beta1, opt.beta2, opt.epsilon, false, nil)
			}
		} else {
			// TODO(liwen)
			return fmt.Errorf("Sparse Adam not implemented")
		}
	}
	return nil
}

// NewAdamOptimizer creates a Adam optimizer instance
func NewAdamOptimizer(lr float32, beta1 float32, beta2 float32, epsilon float32, amsgrad bool) *AdamOptimizer {
	var opt AdamOptimizer
	opt.lr = lr
	opt.step = 0
	opt.beta1 = beta1
	opt.beta2 = beta2
	opt.epsilon = epsilon
	opt.amsgrad = amsgrad
	opt.m = NewParameter()
	opt.v = NewParameter()
	opt.maxSquare = NewParameter()
	return &opt
}

// InitEmbedding set m,v,maxSquare embedding of AdamOptimizer (TODO)
func (opt *AdamOptimizer) InitEmbedding(name string, embedding *common.EmbeddingTable) {
	fmt.Println("unimplemented")
}

// InitNonEmbedding set m,v,maxSquare non-embedding of AdamOptimizer
func (opt *AdamOptimizer) InitNonEmbedding(name string, dim []int64) {
	var size int64 = 1
	for _, d := range dim {
		size *= d
	}
	opt.m.NonEmbeddingParam[name] = &common.Tensor{name, make([]float32, size, size), dim, nil}
	opt.v.NonEmbeddingParam[name] = &common.Tensor{name, make([]float32, size, size), dim, nil}
	opt.maxSquare.NonEmbeddingParam[name] = &common.Tensor{name, make([]float32, size, size), dim, nil}
}
