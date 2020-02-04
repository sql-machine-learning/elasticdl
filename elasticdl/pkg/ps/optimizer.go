package ps

import "elasticdl.org/elasticdl/pkg/common"
import "elasticdl.org/elasticdl/pkg/kernel"
import "fmt"

// Optimizer interface
type Optimizer interface {
	GetLR() float32
	ApplyGradients([]*common.Tensor, *Parameter) error
}

// BaseOptimizer struct
type BaseOptimizer struct {
	lr    float32
	step  int64
	dtype common.DataType
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
func (opt *SGDOptimizer) ApplyGradients(grads []*common.Tensor, p *Parameter) error {
	for _, grad := range grads {
		if grad.Indices == nil {
			t := p.GetNonEmbeddingParam(grad.Name)
			if t == nil {
				return fmt.Errorf("grad %s not in Parameter", grad.Name)
			}
			kernel.SGD(grad.Data, t.Data, opt.GetLR())
		} else {
			t := p.GetEmbeddingParam(grad.Name)
			if t == nil {
				return fmt.Errorf("grad %s not in Parameter", grad.Name)
			}
			kernel.SparseSGD(grad, t, opt.GetLR())
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

// GetLR returns learning rate Adam
func (opt *AdamOptimizer) GetLR() float32 {
	return opt.lr
}

// ApplyGradients applies gradients to parameters
func (opt *AdamOptimizer) ApplyGradients(grads []*common.Tensor, p *Parameter) error {
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
				kernel.Adam(grad.Data, t.Data, m.Data, v.Data, opt.lr, opt.step, opt.beta1, opt.beta2, opt.epsilon, true, ms.Data)
			} else {
				kernel.Adam(grad.Data, t.Data, m.Data, v.Data, opt.lr, opt.step, opt.beta1, opt.beta2, opt.epsilon, false, nil)
			}
		} else {
			t := p.GetEmbeddingParam(grad.Name)
			m := opt.m.GetEmbeddingParam(grad.Name)
			v := opt.v.GetEmbeddingParam(grad.Name)
			if t == nil || m == nil || v == nil {
				return fmt.Errorf("grad %s not in Parameter", grad.Name)
			}
			if opt.amsgrad {
				ms := opt.maxSquare.GetEmbeddingParam(grad.Name)
				kernel.SparseAdam(grad, t, m, v, opt.lr, opt.step, opt.beta1, opt.beta2, opt.epsilon, true, ms)
			} else {
				kernel.SparseAdam(grad, t, m, v, opt.lr, opt.step, opt.beta1, opt.beta2, opt.epsilon, false, nil)
			}
		}
	}
	return nil
}

// NewAdamOptimizer creates a Adam optimizer instance
func NewAdamOptimizer(lr float32, beta1 float32, beta2 float32, epsilon float32, amsgrad bool, dtype common.DataType) *AdamOptimizer {
	var opt = AdamOptimizer{
		BaseOptimizer: BaseOptimizer{
			lr:    lr,
			step:  0,
			dtype: dtype,
		},
		beta1:     beta1,
		beta2:     beta2,
		epsilon:   epsilon,
		amsgrad:   amsgrad,
		m:         NewParameter(dtype),
		v:         NewParameter(dtype),
		maxSquare: NewParameter(dtype),
	}
	return &opt
}

// InitEmbeddingParam set m,v,maxSquare embedding of AdamOptimizer
func (opt *AdamOptimizer) InitEmbeddingParam(name string, dim int64) {
	opt.m.SetEmbeddingParamInfo(name, dim, "zero")
	opt.v.SetEmbeddingParamInfo(name, dim, "zero")
	opt.maxSquare.SetEmbeddingParamInfo(name, dim, "zero")
}

// InitNonEmbeddingParam set m,v,maxSquare non-embedding of AdamOptimizer
func (opt *AdamOptimizer) InitNonEmbeddingParam(name string, dim []int64) {
	opt.m.NonEmbeddingParam[name] = common.NewEmptyTensor(name, opt.dtype, dim...)
	opt.v.NonEmbeddingParam[name] = common.NewEmptyTensor(name, common.Float32Dtype, dim...)
	opt.maxSquare.NonEmbeddingParam[name] = common.NewEmptyTensor(name, common.Float32Dtype, dim...)
}
