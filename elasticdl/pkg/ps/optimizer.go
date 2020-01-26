package ps

import "elasticdl.org/elasticdl/pkg/common"
import "elasticdl.org/elasticdl/pkg/kernel"
import "fmt"

// Optimizer interface
type Optimizer interface {
	GetLR() float32
	ApplyGradients([]common.Tensor, *Parameter) error
}

// Optimizer struct
type Optimizer struct {
	lr float32
}

// SGDOptimizer struct
type SGDOptimizer struct {
	Optimizer
}

// GetLR returns learning rate SGD
func (opt *SGDOptimizer) GetLR() float32 {
	return opt.Optimizer.lr
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

// AdamOptimizer struct
type AdamOptimizer struct {
	Optimizer
	beta1 float32
	beta2 float32
	epsilon float32
	amsgrad bool
	m Parameter
	v Parameter
	maxSquare Parameter
}

// ApplyGradients applies gradients to parameters
func (opt *AdamOptimizer) ApplyGradients(grads []common.Tensor, p *Parameter) error {
	for _, grad := range grads {
		if grad.Indices == nil {
			return fmt.Errorf("Sparse Adam is not implemented")
		} else {
			t := p.GetEmbeddingParam(grad.Name)
			m := opt.m.GetEmbeddingParam(grad.Name)
			v := opt.v.GetEmbeddingParam(grad.Name)
			if t == nil || m == nil || v == nil {
				return fmt.Errorf("grad %s not in Parameter", grad.Name)
			}
			if opt.amsgrad{
				ms := opt.maxSquare.GetEmbeddingParam(grad.Name)
				kernel.Adam(&grad, t, &m, &v, opt.Optimizer.lr, opt.step, opt.beta1, opt.beta2, true, &ms)
			} else {
				kernel.Adam(&grad, t, &m, &v, opt.Optimizer.lr, opt.step, opt.beta1, opt.beta2, false, nil)
			}
		}
	}
	return nil
}

// NewAdamOptimizer creates a Adam optimizer instance
func NewAdamOptimizer(lr float32, beta1 float32, beta2 float32, epsilon float32, amsgrad bool) *SGDOptimizer {
	var opt AdamOptimizer
	opt.Optimizer.lr = lr
	opt.beta1 = beta1
	opt.beta2 = beta2
	opt.epsilon = epsilon
	opt.amsgrad = amsgrad
	return &opt
}