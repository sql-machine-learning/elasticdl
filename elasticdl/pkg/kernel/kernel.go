package kernel

// #cgo LDFLAGS: -L./capi -lkernel_api -lm
// #include "capi/kernel_api.h"
import "C"
import "unsafe"

// SGD kernel
func SGD(grad []float32, param []float32, lr float64, size int64) {
	gradPtr := (*C.float)(unsafe.Pointer(&grad[0]))
	paramPtr := (*C.float)(unsafe.Pointer(&param[0]))
	C.SGD(gradPtr, paramPtr, C.double(lr), C.longlong(size))
}

// Adam kernel
func Adam(grad []float32, param []float32, m []float32, v []float32,
	lr float64, size int64, step int64, beta1 float64, beta2 float64, epsilon float64, amsgrad bool, maxSquare []float32) {
	gradPtr := (*C.float)(unsafe.Pointer(&grad[0]))
	paramPtr := (*C.float)(unsafe.Pointer(&param[0]))
	mPtr := (*C.float)(unsafe.Pointer(&m[0]))
	vPtr := (*C.float)(unsafe.Pointer(&v[0]))
	if amsgrad {
		maxSquarePtr := (*C.float)(unsafe.Pointer(&maxSquare[0]))
		C.Adam(gradPtr, paramPtr, mPtr, vPtr, C.double(lr), C.longlong(size), C.longlong(step),
			C.double(beta1), C.double(beta2), C.double(epsilon), maxSquarePtr)
	} else {
		C.Adam(gradPtr, paramPtr, mPtr, vPtr, C.double(lr), C.longlong(size), C.longlong(step),
			C.double(beta1), C.double(beta2), C.double(epsilon), nil)
	}

}
