package kernel

// #cgo LDFLAGS: -L./capi -lkernel_api
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
