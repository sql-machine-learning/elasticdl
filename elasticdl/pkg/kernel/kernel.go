package kernel

// #cgo LDFLAGS: -L./capi -lkernel_api -lm
// #include "capi/kernel_api.h"
import "C"
import "unsafe"
import "elasticdl.org/elasticdl/pkg/common"
import "fmt"

// SGD kernel
func SGD(grad *common.Vector, param *common.Vector, lr float32) error {
	if grad.Length != param.Length {
		return fmt.Errorf("grad Value size not equal to param")
	}
	gradPtr := unsafe.Pointer(&grad.Data[0])
	paramPtr := unsafe.Pointer(&param.Data[0])
	C.SGD(gradPtr, paramPtr, C.float(lr), C.longlong(grad.Length), C.int(grad.Dtype.Flag))
	return nil
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
		subgrad := grad.RowRef(i)
		SGD(subgrad, vector, lr)
	}
	return nil
}

// Adam kernel
func Adam(grad *common.Vector, param *common.Vector, m *common.Vector, v *common.Vector,
	lr float32, step int64, beta1 float32, beta2 float32,
	epsilon float32, amsgrad bool, maxSquare *common.Vector) {
	gradPtr := unsafe.Pointer(&grad.Data[0])
	paramPtr := unsafe.Pointer(&param.Data[0])
	mPtr := unsafe.Pointer(&m.Data[0])
	vPtr := unsafe.Pointer(&v.Data[0])
	if amsgrad {
		maxSquarePtr := unsafe.Pointer(&maxSquare.Data[0])
		C.Adam(gradPtr, paramPtr, mPtr, vPtr, C.float(lr), C.longlong(grad.Length),
			C.longlong(step), C.float(beta1), C.float(beta2), C.float(epsilon),
			maxSquarePtr, C.int(grad.Dtype.Flag))
	} else {
		C.Adam(gradPtr, paramPtr, mPtr, vPtr, C.float(lr), C.longlong(grad.Length),
			C.longlong(step), C.float(beta1), C.float(beta2), C.float(epsilon), nil,
			C.int(grad.Dtype.Flag))
	}
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
		subgrad := grad.RowRef(i)
		subparam := param.GetEmbeddingVector(index)
		subm := m.GetEmbeddingVector(index)
		subv := v.GetEmbeddingVector(index)
		var submaxs *common.Vector = nil
		if amsgrad {
			submaxs = maxSquare.GetEmbeddingVector(index)
		}
		Adam(subgrad, subparam, subm, subv, lr, step, beta1, beta2, epsilon, amsgrad, submaxs)
	}
	return nil
}
