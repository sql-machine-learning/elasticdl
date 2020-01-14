package kernel

// #cgo CFLAGS: -I/usr/local/include/eigen3
// #cgo LDFLAGS: -L . -lkernel
// #include "kernel.h"
import "C"
import "unsafe"

func SGD(grad []float32, param []float32, lr float64, size int64) {
    gradPtr := (*C.float)(unsafe.Pointer(&grad[0]))
    paramPtr := (*C.float)(unsafe.Pointer(&param[0]))
    C.SGD(gradPtr, paramPtr, C.double(lr), C.longlong(size))
}