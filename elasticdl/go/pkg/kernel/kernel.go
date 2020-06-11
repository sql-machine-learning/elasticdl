// Copyright 2020 The SQLFlow Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kernel

// #cgo LDFLAGS: -L./capi -lkernel_api -lm
// #include "capi/kernel_api.h"
import "C"
import (
	"fmt"
	"unsafe"

	"elasticdl.org/elasticdl/pkg/common"
)

// SGD kernel
func SGD(grad *common.Tensor, param *common.Tensor, lr float32) {
	gradPtr := (*C.float)(unsafe.Pointer(&grad.Buffer[0]))
	paramPtr := (*C.float)(unsafe.Pointer(&param.Buffer[0]))
	length := len(grad.Buffer) / int(common.DtypeSize[grad.Dtype])
	C.SGD(gradPtr, paramPtr, C.float(lr), C.longlong(length))
}

// SparseSGD kernel
func SparseSGD(grad *common.IndexedSlices, param *common.EmbeddingTable, lr float32) error {
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
func IndexedSGD(grad *common.IndexedSlices, param *common.Tensor, lr float32) error {
	for i, index := range grad.Ids {
		vector := param.GetRow(index)
		subGrad := grad.ConcatTensors.GetRow(int64(i))
		SGD(subGrad, vector, lr)
	}
	return nil
}

// Momentum kernel
func Momentum(grad *common.Tensor, param *common.Tensor, velocity *common.Tensor,
	mu float32, nesterov bool, lr float32) {
	gradPtr := (*C.float)(unsafe.Pointer(&grad.Buffer[0]))
	paramPtr := (*C.float)(unsafe.Pointer(&param.Buffer[0]))
	velocityPtr := (*C.float)(unsafe.Pointer(&velocity.Buffer[0]))
	length := len(grad.Buffer) / int(common.DtypeSize[grad.Dtype])
	C.Momentum(gradPtr, paramPtr, velocityPtr, C.float(mu), C._Bool(nesterov),
		C.float(lr), C.longlong(length))
}

// SparseMomentum kernel
func SparseMomentum(grad *common.IndexedSlices, param *common.EmbeddingTable,
	velocity *common.EmbeddingTable, mu float32, nesterov bool, lr float32) error {
	if grad.ConcatTensors.Dims[1] != param.Dim {
		return fmt.Errorf("grad width is not equal to embedding dim")
	}
	for i, index := range grad.Ids {
		vector := param.GetEmbeddingVector(index)
		subGrad := grad.ConcatTensors.GetRow(int64(i))
		subVelocity := velocity.GetEmbeddingVector(index)
		Momentum(subGrad, vector, subVelocity, mu, nesterov, lr)
	}
	return nil
}

// IndexedMomentum kernel
func IndexedMomentum(grad *common.IndexedSlices, param *common.Tensor, velocity *common.Tensor,
	mu float32, nesterov bool, lr float32) error {
	if grad.ConcatTensors.Dims[1] != param.Dims[1] {
		return fmt.Errorf("grad width is not equal to embedding dim")
	}
	for i, index := range grad.Ids {
		vector := param.GetRow(index)
		subGrad := grad.ConcatTensors.GetRow(int64(i))
		subVelocity := velocity.GetRow(index)
		Momentum(subGrad, vector, subVelocity, mu, nesterov, lr)
	}
	return nil
}

// Adam kernel
func Adam(grad *common.Tensor, param *common.Tensor, m *common.Tensor, v *common.Tensor,
	lr float32, step int64, beta1 float32, beta2 float32,
	epsilon float32, amsgrad bool, maxSquare *common.Tensor) {
	gradPtr := (*C.float)(unsafe.Pointer(&grad.Buffer[0]))
	paramPtr := (*C.float)(unsafe.Pointer(&param.Buffer[0]))
	mPtr := (*C.float)(unsafe.Pointer(&m.Buffer[0]))
	vPtr := (*C.float)(unsafe.Pointer(&v.Buffer[0]))
	length := len(grad.Buffer) / int(common.DtypeSize[grad.Dtype])
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
func SparseAdam(grad *common.IndexedSlices, param *common.EmbeddingTable,
	m *common.EmbeddingTable, v *common.EmbeddingTable, lr float32,
	step int64, beta1 float32, beta2 float32, epsilon float32, amsgrad bool,
	maxSquare *common.EmbeddingTable) error {
	if grad.ConcatTensors.Dims[1] != param.Dim {
		return fmt.Errorf("grad width is not equal to embedding dim")
	}
	for i, index := range grad.Ids {
		subgrad := grad.ConcatTensors.GetRow(int64(i))
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

// IndexedAdam kernel
func IndexedAdam(grad *common.IndexedSlices, param *common.Tensor,
	m *common.Tensor, v *common.Tensor, lr float32, step int64,
	beta1 float32, beta2 float32, epsilon float32, amsgrad bool,
	maxSquare *common.Tensor) error {
	if grad.ConcatTensors.Dims[1] != param.Dims[1] {
		return fmt.Errorf("grad width is not equal to embedding dim")
	}
	for i, index := range grad.Ids {
		subgrad := grad.ConcatTensors.GetRow(int64(i))
		subparam := param.GetRow(index)
		subm := m.GetRow(index)
		subv := v.GetRow(index)
		var submaxs *common.Tensor = nil
		if amsgrad {
			submaxs = maxSquare.GetRow(index)
		}
		Adam(subgrad, subparam, subm, subv, lr, step, beta1, beta2, epsilon, amsgrad, submaxs)
	}
	return nil
}

// Adagrad kernel
func Adagrad(grad *common.Tensor, param *common.Tensor, m *common.Tensor, lr float32, epsilon float32) {
	gradPtr := (*C.float)(unsafe.Pointer(&grad.Buffer[0]))
	paramPtr := (*C.float)(unsafe.Pointer(&param.Buffer[0]))
	mPtr := (*C.float)(unsafe.Pointer(&m.Buffer[0]))
	length := len(grad.Buffer) / int(common.DtypeSize[grad.Dtype])
	C.Adagrad(gradPtr, paramPtr, mPtr, C.float(lr), C.longlong(length), C.float(epsilon))
}

// SparseAdagrad kernel
func SparseAdagrad(grad *common.IndexedSlices, param *common.EmbeddingTable,
	m *common.EmbeddingTable, lr float32, epsilon float32) error {
	if grad.ConcatTensors.Dims[1] != param.Dim {
		return fmt.Errorf("grad width is not equal to embedding dim")
	}
	for i, index := range grad.Ids {
		subgrad := grad.ConcatTensors.GetRow(int64(i))
		subparam := param.GetEmbeddingVector(index)
		subm := m.GetEmbeddingVector(index)
		Adagrad(subgrad, subparam, subm, lr, epsilon)
	}
	return nil
}

// IndexedAdagrad kernel
func IndexedAdagrad(grad *common.IndexedSlices, param *common.Tensor,
	m *common.Tensor, lr float32, epsilon float32) error {
	if grad.ConcatTensors.Dims[1] != param.Dims[1] {
		return fmt.Errorf("grad width is not equal to embedding dim")
	}
	for i, index := range grad.Ids {
		subgrad := grad.ConcatTensors.GetRow(int64(i))
		subparam := param.GetRow(index)
		subm := m.GetRow(index)
		Adagrad(subgrad, subparam, subm, lr, epsilon)
	}
	return nil
}
