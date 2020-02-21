package ps

import (
	"elasticdl.org/elasticdl/pkg/commonnew"
	"elasticdl.org/elasticdl/pkg/kernelnew"
	"elasticdl.org/elasticdl/pkg/proto"
	"fmt"
	"strconv"
	"strings"
)

// Optimizer interface
type Optimizer interface {
	GetLR() float32
	ApplyGradients(*proto.Model, *Model) error
	InitOptimizer(*proto.Model) error
}

// BaseOptimizer struct
type BaseOptimizer struct {
	lr            float32
	step          int64
	DenseKernel   func(*commonnew.Tensor, *commonnew.Tensor, string) error
	SparseKernel  func(*commonnew.IndexedSlices, *commonnew.EmbeddingTable, string) error
	IndexedKernel func(*commonnew.IndexedSlices, *commonnew.Tensor, string) error
}

// ApplyGradients base method
func (opt *BaseOptimizer) ApplyGradients(grads *proto.Model, model *Model) error {
	opt.step++
	for name, tensorPB := range grads.DenseParameters {
		grad := commonnew.DeserializeFromTensorProto(tensorPB)
		param := model.GetDenseParameter(name)
		if param == nil {
			return fmt.Errorf("grad %s not in Parameter", name)
		}
		opt.DenseKernel(grad, param, name)
	}
	for name, indexedSlicePB := range grads.EmbeddingTables {
		grad := commonnew.DeserializeFromIndexedSliceProto(indexedSlicePB)
		param := model.GetDenseParameter(name)
		if param == nil {
			table := model.GetEmbeddingTable(name)
			if table == nil {
				return fmt.Errorf("grad %s not in Parameter", name)
			}
			opt.SparseKernel(grad, table, name)
		} else {
			opt.IndexedKernel(grad, param, name)
		}
	}
	return nil
}

// SGDOptimizer struct
type SGDOptimizer struct {
	BaseOptimizer
}

// GetLR returns learning rate
func (opt *SGDOptimizer) GetLR() float32 {
	return opt.lr
}

// NewSGDOptimizer creates a SGD optimizer instance
func NewSGDOptimizer(lr float32) *SGDOptimizer {
	var opt = SGDOptimizer{
		BaseOptimizer: BaseOptimizer{
			lr: lr,
		},
	}
	opt.DenseKernel = func(grad *commonnew.Tensor, param *commonnew.Tensor, name string) error {
		return kernelnew.SGD(grad, param, opt.GetLR())
	}
	opt.SparseKernel = func(grad *commonnew.IndexedSlices, param *commonnew.EmbeddingTable, name string) error {
		return kernelnew.SparseSGD(grad, param, opt.GetLR())
	}
	opt.IndexedKernel = func(grad *commonnew.IndexedSlices, param *commonnew.Tensor, name string) error {
		return kernelnew.IndexedSGD(grad, param, opt.GetLR())
	}
	return &opt
}

// InitOptimizer SGD Nothing to Init
func (opt *SGDOptimizer) InitOptimizer(pb *proto.Model) error {
	return nil
}

// AdamOptimizer struct
type AdamOptimizer struct {
	BaseOptimizer
	beta1     float32
	beta2     float32
	epsilon   float32
	amsgrad   bool
	m         *Model
	v         *Model
	maxSquare *Model
}

// GetLR returns learning rate
func (opt *AdamOptimizer) GetLR() float32 {
	return opt.lr
}

// NewAdamOptimizer creates a Adam optimizer instance
func NewAdamOptimizer(lr float32, beta1 float32, beta2 float32, epsilon float32, amsgrad bool) *AdamOptimizer {
	var opt AdamOptimizer = AdamOptimizer{
		BaseOptimizer: BaseOptimizer{
			lr:   lr,
			step: 0,
		},
		beta1:     beta1,
		beta2:     beta2,
		epsilon:   epsilon,
		amsgrad:   amsgrad,
		m:         NewModel(),
		v:         NewModel(),
		maxSquare: NewModel(),
	}
	opt.DenseKernel = func(grad *commonnew.Tensor, param *commonnew.Tensor, name string) error {
		m := opt.m.GetDenseParameter(name)
		v := opt.v.GetDenseParameter(name)
		if opt.amsgrad {
			ms := opt.maxSquare.GetDenseParameter(name)
			return kernelnew.Adam(grad, param, m, v, opt.GetLR(), opt.step,
				opt.beta1, opt.beta2, opt.epsilon, true, ms)
		}
		return kernelnew.Adam(grad, param, m, v, opt.GetLR(), opt.step,
			opt.beta1, opt.beta2, opt.epsilon, false, nil)
	}
	opt.SparseKernel = func(grad *commonnew.IndexedSlices, param *commonnew.EmbeddingTable, name string) error {
		m := opt.m.GetEmbeddingTable(name)
		v := opt.v.GetEmbeddingTable(name)
		if opt.amsgrad {
			ms := opt.maxSquare.GetEmbeddingTable(name)
			return kernelnew.SparseAdam(grad, param, m, v, opt.GetLR(), opt.step,
				opt.beta1, opt.beta2, opt.epsilon, true, ms)
		}
		return kernelnew.SparseAdam(grad, param, m, v, opt.GetLR(), opt.step,
			opt.beta1, opt.beta2, opt.epsilon, false, nil)
	}
	opt.IndexedKernel = func(grad *commonnew.IndexedSlices, param *commonnew.Tensor, name string) error {
		m := opt.m.GetDenseParameter(name)
		v := opt.v.GetDenseParameter(name)
		if opt.amsgrad {
			ms := opt.maxSquare.GetDenseParameter(name)
			return kernelnew.IndexedAdam(grad, param, m, v, opt.GetLR(), opt.step,
				opt.beta1, opt.beta2, opt.epsilon, true, ms)
		}
		return kernelnew.IndexedAdam(grad, param, m, v, opt.GetLR(), opt.step,
			opt.beta1, opt.beta2, opt.epsilon, false, nil)
	}
	return &opt
}

// InitOptimizer set m,v,maxSquare non-embedding of AdamOptimizer
func (opt *AdamOptimizer) InitOptimizer(pb *proto.Model) error {
	for name, tensor := range pb.DenseParameters {
		dims := commonnew.GetDimFromTensorProto(tensor)
		dtype := tensor.Dtype
		opt.m.DenseParameters[name] = commonnew.NewEmptyTensor(dims, dtype)
		opt.v.DenseParameters[name] = commonnew.NewEmptyTensor(dims, dtype)
		if opt.amsgrad {
			opt.maxSquare.DenseParameters[name] = commonnew.NewEmptyTensor(dims, dtype)
		}
	}
	for _, info := range pb.EmbeddingTableInfo {
		opt.m.SetEmbeddingTableInfo(info)
		opt.v.SetEmbeddingTableInfo(info)
		opt.maxSquare.SetEmbeddingTableInfo(info)
	}
	return nil
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

// parseOptArgs parses optimizer arguments according to optimizer type
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

// NewOptimizer creates optimizer according to optimizer type and arguments
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
