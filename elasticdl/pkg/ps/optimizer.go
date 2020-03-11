package ps

import (
	"elasticdl.org/elasticdl/common"
	"elasticdl.org/elasticdl/kernel"
	"elasticdl.org/elasticdl/proto"
	"fmt"
	"strconv"
	"strings"
)

// Optimizer interface
type Optimizer interface {
	GetLR() float32
	InitOptimizer(*proto.Model)
	ApplyGradients(*proto.Model, *Model, float32) error
}

// BaseOptimizer struct
type BaseOptimizer struct {
	lr            float32
	step          int64
	DenseKernel   func(*common.Tensor, *common.Tensor, string, float32)
	SparseKernel  func(*common.IndexedSlices, *common.EmbeddingTable, string, float32) error
	IndexedKernel func(*common.IndexedSlices, *common.Tensor, string, float32) error
}

// ApplyGradients base method
func (opt *BaseOptimizer) ApplyGradients(grads *proto.Model, model *Model, lrMultiplier float32) error {
	opt.step++
	for name, tensorPB := range grads.DenseParameters {
		grad := common.DeserializeFromTensorProto(tensorPB)
		param := model.GetDenseParameter(name)
		if param == nil {
			return fmt.Errorf("grad %s not in Parameter", name)
		}
		opt.DenseKernel(grad, param, name, lrMultiplier)
	}
	for name, indexedSlicePB := range grads.EmbeddingTables {
		grad := common.DeserializeFromIndexedSliceProto(indexedSlicePB)
		param := model.GetDenseParameter(name)
		if param == nil {
			table := model.GetEmbeddingTable(name)
			if table == nil {
				return fmt.Errorf("grad %s not in Parameter", name)
			}
			err := opt.SparseKernel(grad, table, name, lrMultiplier)
			if err != nil {
				return err
			}
		} else {
			err := opt.IndexedKernel(grad, param, name, lrMultiplier)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// GetLR returns learning rate
func (opt *BaseOptimizer) GetLR() float32 {
	return opt.lr
}

// SGDOptimizer struct
type SGDOptimizer struct {
	BaseOptimizer
}

// NewSGDOptimizer creates a SGD optimizer instance
func NewSGDOptimizer(lr float32) *SGDOptimizer {
	var opt = SGDOptimizer{
		BaseOptimizer: BaseOptimizer{
			lr: lr,
		},
	}
	opt.DenseKernel = func(grad *common.Tensor, param *common.Tensor, name string, lrMultiplier float32) {
		kernel.SGD(grad, param, opt.GetLR()*lrMultiplier)
	}
	opt.SparseKernel = func(grad *common.IndexedSlices, param *common.EmbeddingTable, name string, lrMultiplier float32) error {
		return kernel.SparseSGD(grad, param, opt.GetLR()*lrMultiplier)
	}
	opt.IndexedKernel = func(grad *common.IndexedSlices, param *common.Tensor, name string, lrMultiplier float32) error {
		return kernel.IndexedSGD(grad, param, opt.GetLR()*lrMultiplier)
	}
	return &opt
}

// InitOptimizer SGD Nothing to Init
func (opt *SGDOptimizer) InitOptimizer(pb *proto.Model) {
	return
}

// MomentumOptimizer struct
type MomentumOptimizer struct {
	BaseOptimizer
	mu       float32
	nesterov bool
	v        *Model
}

// NewMomentumOptimizer creates a Momentum optimizer instance
func NewMomentumOptimizer(lr float32, mu float32, nesterov bool) *MomentumOptimizer {
	var opt = MomentumOptimizer{
		BaseOptimizer: BaseOptimizer{
			lr: lr,
		},
		mu:       mu,
		nesterov: nesterov,
		v:        NewModel(),
	}
	opt.DenseKernel = func(grad *common.Tensor, param *common.Tensor, name string, lrMultiplier float32) {
		v := opt.v.GetDenseParameter(name)
		kernel.Momentum(grad, param, v, opt.mu, opt.nesterov, opt.GetLR()*lrMultiplier)
	}
	opt.SparseKernel = func(grad *common.IndexedSlices, param *common.EmbeddingTable, name string,
		lrMultiplier float32) error {
		v := opt.v.GetEmbeddingTable(name)
		return kernel.SparseMomentum(grad, param, v, opt.mu, opt.nesterov, opt.GetLR()*lrMultiplier)
	}
	opt.IndexedKernel = func(grad *common.IndexedSlices, param *common.Tensor, name string,
		lrMultiplier float32) error {
		v := opt.v.GetDenseParameter(name)
		return kernel.IndexedMomentum(grad, param, v, opt.mu, opt.nesterov, opt.GetLR()*lrMultiplier)
	}
	return &opt
}

// InitOptimizer set v non-embedding of MomentumOptimizer
func (opt *MomentumOptimizer) InitOptimizer(pb *proto.Model) {
	for name, tensor := range pb.DenseParameters {
		dims := common.GetDimFromTensorProto(tensor)
		dtype := tensor.Dtype
		opt.v.DenseParameters[name] = common.NewEmptyTensor(dims, dtype)
	}
	for _, info := range pb.EmbeddingTableInfos {
		opt.v.SetEmbeddingTableInfo(info)
	}
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
	opt.DenseKernel = func(grad *common.Tensor, param *common.Tensor, name string, lrMultiplier float32) {
		m := opt.m.GetDenseParameter(name)
		v := opt.v.GetDenseParameter(name)
		if opt.amsgrad {
			ms := opt.maxSquare.GetDenseParameter(name)
			kernel.Adam(grad, param, m, v, opt.GetLR()*lrMultiplier, opt.step,
				opt.beta1, opt.beta2, opt.epsilon, true, ms)
		}
		kernel.Adam(grad, param, m, v, opt.GetLR()*lrMultiplier, opt.step,
			opt.beta1, opt.beta2, opt.epsilon, false, nil)
	}
	opt.SparseKernel = func(grad *common.IndexedSlices, param *common.EmbeddingTable, name string, lrMultiplier float32) error {
		m := opt.m.GetEmbeddingTable(name)
		v := opt.v.GetEmbeddingTable(name)
		if opt.amsgrad {
			ms := opt.maxSquare.GetEmbeddingTable(name)
			return kernel.SparseAdam(grad, param, m, v, opt.GetLR()*lrMultiplier, opt.step,
				opt.beta1, opt.beta2, opt.epsilon, true, ms)
		}
		return kernel.SparseAdam(grad, param, m, v, opt.GetLR()*lrMultiplier, opt.step,
			opt.beta1, opt.beta2, opt.epsilon, false, nil)
	}
	opt.IndexedKernel = func(grad *common.IndexedSlices, param *common.Tensor, name string, lrMultiplier float32) error {
		m := opt.m.GetDenseParameter(name)
		v := opt.v.GetDenseParameter(name)
		if opt.amsgrad {
			ms := opt.maxSquare.GetDenseParameter(name)
			return kernel.IndexedAdam(grad, param, m, v, opt.GetLR()*lrMultiplier, opt.step,
				opt.beta1, opt.beta2, opt.epsilon, true, ms)
		}
		return kernel.IndexedAdam(grad, param, m, v, opt.GetLR()*lrMultiplier, opt.step,
			opt.beta1, opt.beta2, opt.epsilon, false, nil)
	}
	return &opt
}

// InitOptimizer set m,v,maxSquare non-embedding of AdamOptimizer
func (opt *AdamOptimizer) InitOptimizer(pb *proto.Model) {
	for name, tensor := range pb.DenseParameters {
		dims := common.GetDimFromTensorProto(tensor)
		dtype := tensor.Dtype
		opt.m.DenseParameters[name] = common.NewEmptyTensor(dims, dtype)
		opt.v.DenseParameters[name] = common.NewEmptyTensor(dims, dtype)
		if opt.amsgrad {
			opt.maxSquare.DenseParameters[name] = common.NewEmptyTensor(dims, dtype)
		}
	}
	for _, info := range pb.EmbeddingTableInfos {
		opt.m.SetEmbeddingTableInfo(info)
		opt.v.SetEmbeddingTableInfo(info)
		opt.maxSquare.SetEmbeddingTableInfo(info)
	}
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
			return NewMomentumOptimizer(lr, float32(momentum), nesterov), nil
		}
		return NewSGDOptimizer(lr), nil
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
