package kernel

// #cgo LDFLAGS: -L../c -lkernel -lvector -lm
// #include "../c/kernel.h"
import "C"
import "unsafe"
import "elasticdl.org/elasticdl/pkg/common"

// Vector alias
type Vector = common.Vector

// Tensor alias
type Tensor = common.Tensor

// EmbeddingTable alias
type EmbeddingTable = common.EmbeddingTable

// SGD kernel
func SGD(grad *Vector, param *Vector, lr float32) int {
	if grad.Length != param.Length {
		return 1
	}
	C.SGD(unsafe.Pointer(grad), unsafe.Pointer(param), C.float(lr))
	return 0
}

// SparseSGD kernel
func SparseSGD(grad *Tensor, param *EmbeddingTable, lr float32) int {
	if grad.Indices == nil || len(grad.Dim) != 2 {
		return 1
	}
	if grad.Dim[1] != param.Dim {
		return 2
	}
	for i, index := range grad.Indices {
		vector := param.GetEmbeddingVector(index)
		subgrad := grad.RowRef(i)
		SGD(subgrad, vector, lr)
	}
	return 0
}

// Adam kernel
func Adam(grad *Vector, param *Vector, m *Vector, v *Vector,
	lr float32, step int64, beta1 float32, beta2 float32,
	epsilon float32, amsgrad bool, maxSquare *Vector) int {
	if amsgrad {
		C.Adam(unsafe.Pointer(grad), unsafe.Pointer(param), unsafe.Pointer(m), unsafe.Pointer(v), C.float(lr), C.longlong(step), C.float(beta1), C.float(beta2), C.float(epsilon), unsafe.Pointer(maxSquare))
	} else {
		C.Adam(unsafe.Pointer(grad), unsafe.Pointer(param), unsafe.Pointer(m), unsafe.Pointer(v), C.float(lr), C.longlong(step), C.float(beta1), C.float(beta2), C.float(epsilon), nil)
	}
	return 0
}

// SparseAdam kernel
func SparseAdam(grad *common.Tensor, param *common.EmbeddingTable, m *common.EmbeddingTable,
	v *common.EmbeddingTable, lr float32, step int64, beta1 float32, beta2 float32,
	epsilon float32, amsgrad bool, maxSquare *common.EmbeddingTable) int {
	if grad.Indices == nil || len(grad.Dim) != 2 {
		return 1
	}
	if grad.Dim[1] != param.Dim {
		return 2
	}
	for i, index := range grad.Indices {
		subgrad := grad.RowRef(i)
		subparam := param.GetEmbeddingVector(index)
		subm := m.GetEmbeddingVector(index)
		subv := v.GetEmbeddingVector(index)
		var submaxs *Vector = nil
		if amsgrad {
			submaxs = maxSquare.GetEmbeddingVector(index)
		}
		Adam(subgrad, subparam, subm, subv, lr, step, beta1, beta2, epsilon, amsgrad, submaxs)
	}
	return 0
}
