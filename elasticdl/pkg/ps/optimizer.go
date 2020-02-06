package ps

import (
	"elasticdl.org/elasticdl/pkg/common"
	"elasticdl.org/elasticdl/pkg/kernel"
	"fmt"
	"strconv"
	"strings"
)

// Optimizer interface
type Optimizer interface {
	GetLR() float32
	ApplyGradients([]*common.Tensor, *Parameter) error
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
func (opt *SGDOptimizer) ApplyGradients(grads []*common.Tensor, p *Parameter) error {
	for _, grad := range grads {
		nonEmbeddingT := p.GetNonEmbeddingParam(grad.Name)
		if nonEmbeddingT != nil {
			kernel.SGD(grad, nonEmbeddingT, opt.GetLR())
		} else {
			embeddingT := p.GetEmbeddingParam(grad.Name)
			if embeddingT != nil {
				kernel.SparseSGD(grad, embeddingT, opt.GetLR())
			} else {
				return fmt.Errorf("grad %s not in Parameter", grad.Name)
			}
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
		nonEmbeddingT := p.GetNonEmbeddingParam(grad.Name)
		if nonEmbeddingT != nil {
			m := opt.m.GetNonEmbeddingParam(grad.Name)
			v := opt.v.GetNonEmbeddingParam(grad.Name)
			if opt.amsgrad {
				ms := opt.maxSquare.GetNonEmbeddingParam(grad.Name)
				kernel.Adam(grad, nonEmbeddingT, m, v, opt.lr, opt.step,
					opt.beta1, opt.beta2, opt.epsilon, true, ms)
			} else {
				kernel.Adam(grad, nonEmbeddingT, m, v, opt.lr, opt.step,
					opt.beta1, opt.beta2, opt.epsilon, false, nil)
			}
		} else {
			embeddingT := p.GetEmbeddingParam(grad.Name)
			if embeddingT != nil {
				m := opt.m.GetEmbeddingParam(grad.Name)
				v := opt.v.GetEmbeddingParam(grad.Name)
				if opt.amsgrad {
					ms := opt.maxSquare.GetEmbeddingParam(grad.Name)
					kernel.SparseAdam(grad, embeddingT, m, v, opt.lr, opt.step,
						opt.beta1, opt.beta2, opt.epsilon, true, ms)
				} else {
					kernel.SparseAdam(grad, embeddingT, m, v, opt.lr, opt.step,
						opt.beta1, opt.beta2, opt.epsilon, false, nil)
				}
			} else {
				return fmt.Errorf("grad %s not in Parameter", grad.Name)
			}
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

// InitEmbeddingParam set m,v,maxSquare embedding of AdamOptimizer (TODO)
func (opt *AdamOptimizer) InitEmbeddingParam(name string, dim int64) {
	opt.m.SetEmbeddingParamInfo(name, dim, "zero")
	opt.v.SetEmbeddingParamInfo(name, dim, "zero")
	opt.maxSquare.SetEmbeddingParamInfo(name, dim, "zero")
}

// InitNonEmbeddingParam set m,v,maxSquare non-embedding of AdamOptimizer
func (opt *AdamOptimizer) InitNonEmbeddingParam(name string, dim []int64) {
	size := common.GetDimProduct(dim)
	opt.m.NonEmbeddingParam[name] = &common.Tensor{name, make([]float32, size, size), dim, nil}
	opt.v.NonEmbeddingParam[name] = &common.Tensor{name, make([]float32, size, size), dim, nil}
	opt.maxSquare.NonEmbeddingParam[name] = &common.Tensor{name, make([]float32, size, size), dim, nil}
}

const (
	optTypeSGD     = "SGD"
	optTypeAdam    = "Adam"
	optArgLR       = "learning_rate"
	optArgMomentum = "momentum"
	optArgNesterov = "nesterov"
	optArgBeta1    = "beta_1"
	optArgBeta2    = "beta_2"
	optArgEpsilon  = "epsilon"
	optArgAmsgrad  = "amsgrad"
)

var optArgumentsMap = map[string][]string{
	"SGD":  []string{optArgLR, optArgMomentum, optArgNesterov},
	"Adam": []string{optArgLR, optArgBeta1, optArgBeta2, optArgEpsilon, optArgAmsgrad},
}

func parseOptArgs(optType string, optArgs string) (map[string]string, error) {
	// parse arguments to map
	argsMap := make(map[string]string)
	for _, args := range strings.Split(optArgs, ";") {
		if args == "" {
			continue
		} else {
			arr := strings.Split(args, "=")
			argsMap[arr[0]] = arr[1]
		}
	}

	// check argument names
	for _, argName := range optArgumentsMap[optType] {
		if _, ok := argsMap[argName]; !ok {
			return nil, fmt.Errorf("Args passed to ps should contain %s", argName)
		}
	}
	if len(argsMap) != len(optArgumentsMap[optType]) {
		return nil, fmt.Errorf("Args passed to ps contain redundant items: %v", argsMap)
	}
	return argsMap, nil
}

// NewOptimizer creates Optimizer according to optimizer type and arguments
func NewOptimizer(optType string, optArgs string) (Optimizer, error) {
	argsMap, err := parseOptArgs(optType, optArgs)
	if err != nil {
		return nil, err
	}

	lr64, err := strconv.ParseFloat(argsMap[optArgLR], 32)
	if err != nil {
		return nil, fmt.Errorf("Having error converting learning rate to number: %v", err)
	}
	lr := float32(lr64)
	if optType == optTypeSGD {
		var (
			momentum float64
			nesterov bool
		)
		momentum, err = strconv.ParseFloat(argsMap[optArgMomentum], 32)
		if err != nil {
			return nil, err
		}
		nesterov, err = strconv.ParseBool(argsMap[optArgNesterov])
		if err != nil {
			return nil, err
		}

		if momentum > float64(0.0) {
			return nil, fmt.Errorf("SGD optimizer with momentum has not been implemented")
		}
		if !nesterov {
			return NewSGDOptimizer(lr), nil
		}
		return nil, fmt.Errorf("SGD optimizer with nesterov=true has not been implemented")
	} else if optType == optTypeAdam {
		var (
			beta1   float64
			beta2   float64
			epsilon float64
			amsgrad bool
		)
		beta1, err = strconv.ParseFloat(argsMap[optArgBeta1], 32)
		if err != nil {
			return nil, err
		}
		beta2, err = strconv.ParseFloat(argsMap[optArgBeta2], 32)
		if err != nil {
			return nil, err
		}
		epsilon, err = strconv.ParseFloat(argsMap[optArgEpsilon], 32)
		if err != nil {
			return nil, err
		}
		amsgrad, err = strconv.ParseBool(argsMap[optArgAmsgrad])
		if err != nil {
			return nil, err
		}
		return NewAdamOptimizer(lr, float32(beta1), float32(beta2), float32(epsilon), amsgrad), nil
	} else {
		return nil, fmt.Errorf("Unknown optimizer type %s", optType)
	}
}
