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
func Adam(grad []float32, param []float32, m []float32, v []float32,
	lr float64, size int64, step int64, beta1 float64, beta2 float64,
	epsilon float64, amsgrad bool, maxSquare []float32) {
	gradPtr := (*C.float)(unsafe.Pointer(&grad[0]))
	paramPtr := (*C.float)(unsafe.Pointer(&param[0]))
	mPtr := (*C.float)(unsafe.Pointer(&m[0]))
	vPtr := (*C.float)(unsafe.Pointer(&v[0]))
	if amsgrad {
		maxSquarePtr := (*C.float)(unsafe.Pointer(&maxSquare[0]))
		C.Adam(gradPtr, paramPtr, mPtr, vPtr, C.double(lr), C.longlong(size),
			C.longlong(step), C.double(beta1), C.double(beta2), C.double(epsilon),
			maxSquarePtr)
	} else {
		C.Adam(gradPtr, paramPtr, mPtr, vPtr, C.double(lr), C.longlong(size),
			C.longlong(step), C.double(beta1), C.double(beta2), C.double(epsilon), nil)
	}

}
