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

package common

import (
	"fmt"
	"reflect"
	"unsafe"

	"elasticdl.org/elasticdl/pkg/proto"
	"github.com/tensorflow/tensorflow/tensorflow/go/core/framework/tensor_go_proto"
	"github.com/tensorflow/tensorflow/tensorflow/go/core/framework/tensor_shape_go_proto"
	"github.com/tensorflow/tensorflow/tensorflow/go/core/framework/types_go_proto"
)

// Tensor definition
type Tensor struct {
	Buffer []byte
	Dims   []int64
	Dtype  types_go_proto.DataType
}

// NewEmptyTensor create an empty n-dim tensor
func NewEmptyTensor(dim []int64, dtype types_go_proto.DataType) *Tensor {
	return &Tensor{
		Buffer: make([]byte, DimProduct(dim)*int64(DtypeSize[dtype])),
		Dims:   dim,
		Dtype:  dtype,
	}
}

// NewTensor create a n-dim tensor using exsiting slice
func NewTensor(slice interface{}, dim []int64) *Tensor {
	v := reflect.ValueOf(slice)
	length := v.Len()
	dtype := SliceTypeToDtype[reflect.TypeOf(slice)]
	bytelen := length * int(DtypeSize[dtype])
	if int64(length) != DimProduct(dim) {
		return nil
	}
	sliceHeader := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(v.Pointer())),
		Cap:  int(bytelen),
		Len:  int(bytelen),
	}
	return &Tensor{
		Buffer: *(*[]byte)(unsafe.Pointer(&sliceHeader)),
		Dims:   dim,
		Dtype:  dtype,
	}
}

// NewEmptyVector create an empty 1-dim tensor
func NewEmptyVector(dim int64, dtype types_go_proto.DataType) *Tensor {
	return &Tensor{
		Buffer: make([]byte, dim*int64(DtypeSize[dtype])),
		Dims:   []int64{dim},
		Dtype:  dtype,
	}
}

// NewVector create an empty 1-dim tensor
func NewVector(slice interface{}) *Tensor {
	v := reflect.ValueOf(slice)
	length := v.Len()
	dtype := SliceTypeToDtype[reflect.TypeOf(slice)]
	bytelen := length * int(DtypeSize[dtype])
	if v.Len() != length {
		return nil
	}
	sliceHeader := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(v.Pointer())),
		Cap:  int(bytelen),
		Len:  int(bytelen),
	}
	return &Tensor{
		Buffer: *(*[]byte)(unsafe.Pointer(&sliceHeader)),
		Dims:   []int64{int64(length)},
		Dtype:  dtype,
	}
}

// DimProduct get the number of the elements of a tensor of this dim
func DimProduct(dim []int64) int64 {
	var size int64 = 1
	for _, d := range dim {
		size *= d
	}
	return size
}

// GetSubTensor get the part reference of the tensor
func (t *Tensor) GetSubTensor(begin int64, length int64) *Tensor {
	dsize := int64(DtypeSize[t.Dtype])
	begin *= dsize
	return &Tensor{
		Buffer: t.Buffer[begin : begin+length*dsize],
		Dims:   []int64{length},
		Dtype:  t.Dtype,
	}
}

// GetRow get the row reference of a 2-dim tensor
func (t *Tensor) GetRow(idx int64) *Tensor {
	if len(t.Dims) != 2 || idx >= t.Dims[0] {
		return nil
	}
	begin := t.Dims[1] * idx
	return t.GetSubTensor(begin, t.Dims[1])
}

// SetSubTensor set a vector to an index of tensor
func (t *Tensor) SetSubTensor(begin int64, length int64, val *Tensor) {
	dsize := int64(DtypeSize[t.Dtype])
	begin *= dsize
	length *= dsize
	copy(t.Buffer[begin:begin+length], val.Buffer)
}

// SetRow set a vector to an index of tensor
func (t *Tensor) SetRow(idx int64, vec *Tensor) {
	if len(t.Dims) != 2 || idx >= t.Dims[0] {
		return
	}
	begin := t.Dims[1] * idx
	t.SetSubTensor(begin, t.Dims[1], vec)
}

// Slice gives a Slice interface to the Tensor data
func Slice(t *Tensor) interface{} {
	length := int(DimProduct(t.Dims))
	sliceHeader := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&t.Buffer[0])),
		Cap:  length,
		Len:  length,
	}
	val := reflect.NewAt(DtypeToSliceType[t.Dtype], unsafe.Pointer(&sliceHeader)).Elem()
	return val.Interface()
}

// IsValid check tensor validity
func (t *Tensor) IsValid() bool {
	if DimProduct(t.Dims) != int64(len(t.Buffer)/int(DtypeSize[t.Dtype])) {
		return false
	}
	return true
}

// GetDimFromTensorProto get dim from proto
func GetDimFromTensorProto(pb *tensor_go_proto.TensorProto) []int64 {
	pbDim := pb.GetTensorShape().GetDim()
	dims := make([]int64, len(pbDim), len(pbDim))
	for i, iDim := range pbDim {
		dims[i] = iDim.GetSize()
	}
	return dims
}

// DeserializeFromTensorProto transforms pb to tensor
func DeserializeFromTensorProto(pb *tensor_go_proto.TensorProto) *Tensor {
	dims := GetDimFromTensorProto(pb)
	if int(DimProduct(dims))*int(DtypeSize[pb.GetDtype()]) != len(pb.GetTensorContent()) {
		return nil
	}

	return &Tensor{
		Buffer: pb.GetTensorContent(),
		Dims:   dims,
		Dtype:  pb.GetDtype(),
	}
}

// SerializeToTensorProto transforms tensor to pb
func (t *Tensor) SerializeToTensorProto() *tensor_go_proto.TensorProto {
	shapeDim := make([]*tensor_shape_go_proto.TensorShapeProto_Dim, len(t.Dims), len(t.Dims))
	for i, dim := range t.Dims {
		shapeDim[i] = &tensor_shape_go_proto.TensorShapeProto_Dim{
			Size: dim,
		}
	}
	pbDim := tensor_shape_go_proto.TensorShapeProto{
		Dim: shapeDim,
	}
	return &tensor_go_proto.TensorProto{
		TensorContent: t.Buffer,
		TensorShape:   &pbDim,
		Dtype:         t.Dtype,
	}
}

// IndexedSlices : IndexedSlices in-memory representation
type IndexedSlices struct {
	ConcatTensors *Tensor
	Ids           []int64
}

// NewIndexedSlices return a IndexedTensor instance
func NewIndexedSlices(t *Tensor, ids []int64) *IndexedSlices {
	return &IndexedSlices{
		ConcatTensors: t,
		Ids:           ids,
	}
}

// SerializeToIndexedSlicesProto return proto.IndexedSlices
func (t *IndexedSlices) SerializeToIndexedSlicesProto() *proto.IndexedSlicesProto {
	if t.ConcatTensors.Dims[0] != int64(len(t.Ids)) || len(t.ConcatTensors.Dims) != 2 {
		return nil
	}
	return &proto.IndexedSlicesProto{
		ConcatTensors: t.ConcatTensors.SerializeToTensorProto(),
		Ids:           t.Ids,
	}
}

// DeserializeFromIndexedSliceProto return common.IndexedTensor
func DeserializeFromIndexedSliceProto(pb *proto.IndexedSlicesProto) *IndexedSlices {
	return &IndexedSlices{
		ConcatTensors: DeserializeFromTensorProto(pb.ConcatTensors),
		Ids:           pb.Ids,
	}
}

// MergeIndexedSlices merges two indexed slices into one indexed slices
func MergeIndexedSlices(first *IndexedSlices, second *IndexedSlices) (*IndexedSlices, error) {
	if first == nil {
		return second, nil
	}
	if second == nil {
		return first, nil
	}
	if first.ConcatTensors.Dtype != second.ConcatTensors.Dtype {
		return nil, fmt.Errorf("Could not merge two IndexedSlices with different types")
	}
	if first.ConcatTensors.Dims[1] != second.ConcatTensors.Dims[1] {
		return nil, fmt.Errorf("Could not merge two IndexedSlices with different widths")
	}
	height := first.ConcatTensors.Dims[0] + second.ConcatTensors.Dims[0]
	width := first.ConcatTensors.Dims[1]
	dtype := first.ConcatTensors.Dtype
	tensor := NewEmptyTensor([]int64{height, width}, dtype)
	var ids []int64
	for i, id := range first.Ids {
		tensor.SetRow(int64(i), first.ConcatTensors.GetRow(int64(i)))
		ids = append(ids, id)
	}
	start := len(ids)
	for i, id := range second.Ids {
		tensor.SetRow(int64(start+i), second.ConcatTensors.GetRow(int64(i)))
		ids = append(ids, id)
	}
	return NewIndexedSlices(tensor, ids), nil
}
