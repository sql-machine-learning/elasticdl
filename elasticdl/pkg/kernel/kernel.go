package kernel

// #cgo LDFLAGS: -L./capi -lkernel_api
// #include "capi/kernel_api.h"
import "C"
import "unsafe"

func SGD(grad []float32, param []float32, lr float64, size int64) {
    gradPtr := (*C.float)(unsafe.Pointer(&grad[0]))
    paramPtr := (*C.float)(unsafe.Pointer(&param[0]))
    C.SGD(gradPtr, paramPtr, C.double(lr), C.longlong(size))
}
