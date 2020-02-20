package commonnew

import (
	"github.com/tensorflow/tensorflow/tensorflow/go/core/framework/tensor_go_proto"
	"github.com/tensorflow/tensorflow/tensorflow/go/core/framework/tensor_shape_go_proto"
	"github.com/tensorflow/tensorflow/tensorflow/go/core/framework/types_go_proto"
	"reflect"
	"unsafe"
)

// Tensor is an alias for tensor_go_proto.TensorProto of tensorflow
type Tensor tensor_go_proto.TensorProto

// GetDim returns dims of a tensor
func (t *Tensor) GetDim() []int64 {
	pbDim := t.GetTensorShape().GetDim()
	dims := make([]int64, len(pbDim), len(pbDim))
	for i, iDim := range pbDim {
		dims[i] = iDim.GetSize()
	}
	return dims
}

// SetDim sets dims of a tensor
func (t *Tensor) SetDim(dims []int64) {
	t.TensorShape.Dim = make([]*tensor_shape_go_proto.TensorShapeProto_Dim, len(dims), len(dims))
	for i, dim := range dims {
		t.TensorShape.Dim[i].Size = dim
	}
}

// NewEmptyTensor create an empty n-dim tensor
func NewEmptyTensor(dim []int64, dtype types_go_proto.DataType) *Tensor {
	var t Tensor
	t.SetDim(dim)
	t.TensorContent = make([]byte, DimProduct(dim)*int64(DtypeSize[dtype]))
	t.Dtype = dtype
	return &t
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
	var t Tensor
	t.SetDim(dim)
	t.TensorContent = *(*[]byte)(unsafe.Pointer(&sliceHeader))
	t.Dtype = dtype
	return &t
}

// NewEmptyVector create an empty 1-dim tensor
func NewEmptyVector(dim int64, dtype types_go_proto.DataType) *Tensor {
	dims := []int64{dim}
	return NewEmptyTensor(dims, dtype)
}

// NewVector create an empty 1-dim tensor
func NewVector(slice interface{}) *Tensor {
	v := reflect.ValueOf(slice)
	length := v.Len()
	dims = []int64{int64(length)}
	return NewTensor(dims, slice)
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
	var subT Tensor
	subT.SetDim([]int64{length})
	subT.Dtype = t.Dtype
	subT.TensorContent = t.TensorContent[begin : begin+length*dsize]
	return &subT
}

// GetRow get the row reference of a 2-dim tensor
func (t *Tensor) GetRow(idx int64) *Tensor {
	dims = t.GetDim()
	if len(dims) != 2 || idx >= dims[0] {
		return nil
	}
	begin := dims[1] * idx
	return t.GetSubTensor(begin, dims[1])
}

// SetSubTensor set a vector to an index of tensor
func (t *Tensor) SetSubTensor(begin int64, length int64, val *Tensor) {
	dsize := int64(DtypeSize[t.Dtype])
	begin *= dsize
	length *= dsize
	copy(t.TensorContent[begin:begin+length], val.TensorContent)
}

// SetRow set a vector to an index of tensor
func (t *Tensor) SetRow(idx int64, vec *Tensor) {
	dims = t.GetDim()
	if len(dims) != 2 || idx >= dims[0] {
		return
	}
	begin := dims[1] * idx
	t.SetSubTensor(begin, dims[1], vec)
}

// Slice gives a Slice interface to the Tensor data
func Slice(t *Tensor) interface{} {
	length := int(DimProduct(t.GetDim()))
	sliceHeader := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&t.TensorContent[0])),
		Cap:  length,
		Len:  length,
	}
	val := reflect.NewAt(DtypeToSliceType[t.Dtype], unsafe.Pointer(&sliceHeader)).Elem()
	return val.Interface()
}
