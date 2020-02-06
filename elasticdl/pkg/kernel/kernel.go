package kernel

// #cgo LDFLAGS: -L./capi -lkernel_api -lm
// #include "capi/kernel_api.h"
import "C"
import "unsafe"
import "elasticdl.org/elasticdl/pkg/common"
import "fmt"

func sgd(grad *common.Tensor, param *common.Tensor, lr float32) error {
	if len(grad.Value) != len(param.Value) {
		return fmt.Errorf("grad %s Value size not equal to param", grad.Name)
	}
	gradPtr := (*C.float)(unsafe.Pointer(&grad.Value[0]))
	paramPtr := (*C.float)(unsafe.Pointer(&param.Value[0]))
	C.SGD(gradPtr, paramPtr, C.float(lr), C.longlong(len(grad.Value)))
	return nil
}

// SGD kernel
func SGD(grad *common.Tensor, param *common.Tensor, lr float32) error {
	// support tf.IndexedSlices gradient for dense param
	if grad.Indices != nil {
		for i, index := range grad.Indices {
			subVector := param.AtRow(index)
			subGrad := grad.AtRow(int64(i))
			sgd(subVector, subGrad, lr)
		}
		return nil
	}
	return sgd(grad, param, lr)
}

// SparseSGD kernel
func SparseSGD(grad *common.Tensor, param *common.EmbeddingTable, lr float32) error {
	if grad.Indices == nil || len(grad.Dim) != 2 {
		return fmt.Errorf("grad %s is not row sparse tensor", grad.Name)
	}
	if grad.Dim[1] != param.Dim {
		return fmt.Errorf("grad %s width is not equal to embedding dim", grad.Name)
	}
	for i, index := range grad.Indices {
		vector := param.GetEmbeddingVector(index)
		subGrad := grad.AtRow(int64(i))
		sgd(subGrad, vector, lr)
	}
	return nil
}

func adam(grad *common.Tensor, param *common.Tensor, m *common.Tensor, v *common.Tensor,
	lr float32, step int64, beta1 float32, beta2 float32,
	epsilon float32, amsgrad bool, maxSquare *common.Tensor) {
	gradPtr := (*C.float)(unsafe.Pointer(&grad.Value[0]))
	paramPtr := (*C.float)(unsafe.Pointer(&param.Value[0]))
	mPtr := (*C.float)(unsafe.Pointer(&m.Value[0]))
	vPtr := (*C.float)(unsafe.Pointer(&v.Value[0]))
	size := C.longlong(len(grad.Value))
	if amsgrad {
		maxSquarePtr := (*C.float)(unsafe.Pointer(&maxSquare.Value[0]))
		C.Adam(gradPtr, paramPtr, mPtr, vPtr, C.float(lr), C.longlong(size),
			C.longlong(step), C.float(beta1), C.float(beta2), C.float(epsilon),
			maxSquarePtr)
	} else {
		C.Adam(gradPtr, paramPtr, mPtr, vPtr, C.float(lr), C.longlong(size),
			C.longlong(step), C.float(beta1), C.float(beta2), C.float(epsilon), nil)
	}
}

func addTo(a *common.Tensor, b *common.Tensor) {
	aPtr := (*C.float)(unsafe.Pointer(&a.Value[0]))
	bPtr := (*C.float)(unsafe.Pointer(&b.Value[0]))
	size := C.longlong(len(a.Value))
	C.AddTo(aPtr, bPtr, size)
}

func removeDuplicateElement(indices []int64) (map[int64]int64, []int64) {
	result := make(map[int64]int64)
	newIndices := make([]int64, 0, len(indices))
	i := 0
	temp := map[int64]struct{}{}
	for _, item := range indices {
		if _, ok := temp[item]; !ok {
			temp[item] = struct{}{}
			result[item] = int64(i)
			newIndices = append(newIndices, item)
			i++
		}
	}
	return result, newIndices
}

func mergeIndexedSlices(grad *common.Tensor) *common.Tensor {
	result, newIndices := removeDuplicateElement(grad.Indices)
	newD := []int64{int64(len(newIndices)), grad.Dim[1]}
	t := common.NewTensor(newD, grad.Name)
	t.Indices = newIndices
	for i, index := range grad.Indices {
		subA := t.AtRow(result[index])
		subB := grad.AtRow(int64(i))
		addTo(subA, subB)
	}
	return t
}

// Adam kernel
func Adam(grad *common.Tensor, param *common.Tensor, m *common.Tensor, v *common.Tensor,
	lr float32, step int64, beta1 float32, beta2 float32,
	epsilon float32, amsgrad bool, maxSquare *common.Tensor) {
	// support tf.IndexedSlices gradient for dense param
	if grad.Indices != nil {
		newGrad := mergeIndexedSlices(grad)
		for i, index := range newGrad.Indices {
			subGrad := newGrad.AtRow(int64(i))
			subParam := param.AtRow(index)
			subM := m.AtRow(index)
			subV := v.AtRow(index)
			var subMaxs *common.Tensor = nil
			if amsgrad {
				subMaxs = maxSquare.AtRow(index)
			}
			adam(subGrad, subParam, subM, subV, lr, step, beta1, beta2, epsilon, amsgrad, subMaxs)
		}
		return
	}
	adam(grad, param, m, v, lr, step, beta1, beta2, epsilon, amsgrad, maxSquare)
}

// SparseAdam kernel
func SparseAdam(grad *common.Tensor, param *common.EmbeddingTable, m *common.EmbeddingTable,
	v *common.EmbeddingTable, lr float32, step int64, beta1 float32, beta2 float32,
	epsilon float32, amsgrad bool, maxSquare *common.EmbeddingTable) error {
	if grad.Indices == nil || len(grad.Dim) != 2 {
		return fmt.Errorf("grad %s is not row sparse tensor", grad.Name)
	}
	if grad.Dim[1] != param.Dim {
		return fmt.Errorf("grad %s width is not equal to embedding dim", grad.Name)
	}
	for i, index := range grad.Indices {
		subgrad := grad.AtRow(int64(i))
		subparam := param.GetEmbeddingVector(index)
		subm := m.GetEmbeddingVector(index)
		subv := v.GetEmbeddingVector(index)
		var submaxs *common.Tensor = nil
		if amsgrad {
			submaxs = maxSquare.GetEmbeddingVector(index)
		}
		Adam(subgrad, subparam, subm, subv, lr, step, beta1, beta2, epsilon, amsgrad, submaxs)
	}
	return nil
}
