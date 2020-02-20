package kernelnew

// #cgo LDFLAGS: -L./capi -lkernel_api -lm
// #include "capi/kernel_api.h"
import "C"
import (
	"elasticdl.org/elasticdl/pkg/commonnew"
	"fmt"
	"unsafe"
)

// SGD kernel
func SGD(grad *commonnew.Tensor, param *commonnew.Tensor, lr float32) error {
	gradPtr := (*C.float)(unsafe.Pointer(&grad.Buffer[0]))
	paramPtr := (*C.float)(unsafe.Pointer(&param.Buffer[0]))
	length := len(grad.Buffer) / int(commonnew.DtypeSize[grad.Dtype])
	C.SGD(gradPtr, paramPtr, C.float(lr), C.longlong(length))
	return nil
}

// SparseSGD kernel
func SparseSGD(grad *commonnew.IndexedSlices, param *commonnew.EmbeddingTable, lr float32) error {
	if grad.ConcatTensors.Dims[1] != param.Dim {
		return fmt.Errorf("grad width is not equal to embedding dim")
	}
	for i, index := range grad.Ids {
		vector := param.GetEmbeddingVector(index)
		subGrad := grad.ConcatTensors.GetRow(int64(i))
		SGD(subGrad, vector, lr)
	}
	return nil
}

// IndexedSGD kernel
func IndexedSGD(grad *commonnew.IndexedSlices, param *commonnew.Tensor, lr float32) error {
	for i, index := range grad.Ids {
		vector := param.GetRow(index)
		subGrad := grad.ConcatTensors.GetRow(int64(i))
		SGD(subGrad, vector, lr)
	}
	return nil
}

// Adam kernel
func Adam(grad *commonnew.Tensor, param *commonnew.Tensor, m *commonnew.Tensor, v *commonnew.Tensor,
	lr float32, step int64, beta1 float32, beta2 float32,
	epsilon float32, amsgrad bool, maxSquare *commonnew.Tensor) {
	gradPtr := (*C.float)(unsafe.Pointer(&grad.Buffer[0]))
	paramPtr := (*C.float)(unsafe.Pointer(&param.Buffer[0]))
	mPtr := (*C.float)(unsafe.Pointer(&m.Buffer[0]))
	vPtr := (*C.float)(unsafe.Pointer(&v.Buffer[0]))
	length := len(grad.Buffer) / int(commonnew.DtypeSize[grad.Dtype])
	if amsgrad {
		maxSquarePtr := (*C.float)(unsafe.Pointer(&maxSquare.Buffer[0]))
		C.Adam(gradPtr, paramPtr, mPtr, vPtr, C.float(lr), C.longlong(length),
			C.longlong(step), C.float(beta1), C.float(beta2), C.float(epsilon),
			maxSquarePtr)
	} else {
		C.Adam(gradPtr, paramPtr, mPtr, vPtr, C.float(lr), C.longlong(length),
			C.longlong(step), C.float(beta1), C.float(beta2), C.float(epsilon), nil)
	}
}

// SparseAdam kernel
func SparseAdam(grad *commonnew.IndexedSlices, param *commonnew.EmbeddingTable,
	m *commonnew.EmbeddingTable, v *commonnew.EmbeddingTable, lr float32,
	step int64, beta1 float32, beta2 float32, epsilon float32, amsgrad bool,
	maxSquare *commonnew.EmbeddingTable) error {
	if grad.ConcatTensors.Dims[1] != param.Dim {
		return fmt.Errorf("grad width is not equal to embedding dim")
	}
	for i, index := range grad.Ids {
		subgrad := grad.ConcatTensors.GetRow(int64(i))
		subparam := param.GetEmbeddingVector(index)
		subm := m.GetEmbeddingVector(index)
		subv := v.GetEmbeddingVector(index)
		var submaxs *commonnew.Tensor = nil
		if amsgrad {
			submaxs = maxSquare.GetEmbeddingVector(index)
		}
		Adam(subgrad, subparam, subm, subv, lr, step, beta1, beta2, epsilon, amsgrad, submaxs)
	}
	return nil
}

// IndexedAdam kernel
func IndexedAdam(grad *commonnew.IndexedSlices, param *commonnew.Tensor,
	m *commonnew.Tensor, v *commonnew.Tensor, lr float32, step int64,
	beta1 float32, beta2 float32, epsilon float32, amsgrad bool,
	maxSquare *commonnew.Tensor) error {
	if grad.ConcatTensors.Dims[1] != param.Dims[1] {
		return fmt.Errorf("grad width is not equal to embedding dim")
	}
	for i, index := range grad.Ids {
		subgrad := grad.ConcatTensors.GetRow(int64(i))
		subparam := param.GetRow(index)
		subm := m.GetRow(index)
		subv := v.GetRow(index)
		var submaxs *commonnew.Tensor = nil
		if amsgrad {
			submaxs = maxSquare.GetRow(index)
		}
		Adam(subgrad, subparam, subm, subv, lr, step, beta1, beta2, epsilon, amsgrad, submaxs)
	}
	return nil
}
