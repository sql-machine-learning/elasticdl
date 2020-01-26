package kernel

// #cgo LDFLAGS: -L./capi -lkernel_api -lm
// #include "capi/kernel_api.h"
import "C"
import "unsafe"
import "elasticdl.org/elasticdl/pkg/common"
import "fmt"

// SGD kernel
func SGD(grad *common.Tensor, param *common.Tensor, lr float32) error {
	if len(grad.Value) != len(param.Value) {
		return fmt.Errorf("grad %s Value size not equal to param", grad.Name)
	}
	gradPtr := (*C.float)(unsafe.Pointer(&grad.Value[0]))
	paramPtr := (*C.float)(unsafe.Pointer(&param.Value[0]))
	C.SGD(gradPtr, paramPtr, C.float(lr), C.longlong(len(grad.Value)))
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
		for j := range vector {
			vector[j] -= lr * grad.Value[j+i*int(param.Dim)]
		}
	}
	return nil
}

// Adam kernel
func Adam(grad *common.Tensor, param *common.Tensor, m *common.Tensor, v *common.Tensor,
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
