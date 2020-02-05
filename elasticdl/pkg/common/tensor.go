package common

import (
	"bytes"
	"elasticdl.org/elasticdl/pkg/proto"
	"encoding/binary"
	"fmt"
)

// Tensor defines tensor struct
type Tensor struct {
	Name    string
	Data    *Vector
	Dim     []int64
	Indices []int64
}

// ProductInt64 get the number of the elements of a tensor of this dim
func ProductInt64(dim []int64) int64 {
	var size int64 = 1
	for _, d := range dim {
		size *= d
	}
	return size
}

// NewEmptyTensor create a new n-dim tensor
func NewEmptyTensor(name string, dim []int64, dtype DataType) *Tensor {
	var t = Tensor{
		Name: name,
		Data: NewEmptyVector(int(ProductInt64(dim)), dtype),
		Dim:  dim,
	}
	return &t
}

// NewInitializedTensor create an initialized vector
func NewInitializedTensor(name string, dim []int64, indices []int64, dtype DataType, initializer InitializeFunc) *Tensor {
	var t = Tensor{
		Name: name,
		Data: NewInitializedVector(int(ProductInt64(dim)), dtype, initializer),
		Dim:  dim,
	}
	return &t
}

// NewTensor create tensor from slice
func NewTensor(name string, data interface{}, dim []int64, indices []int64) *Tensor {
	var t = Tensor{
		Name:    name,
		Data:    NewVector(data),
		Dim:     dim,
		Indices: indices,
	}
	return &t
}

// NewTensorInplace create tensor from slice inplace
func NewTensorInplace(name string, data interface{}, dim []int64, indices []int64) *Tensor {
	var t = Tensor{
		Name:    name,
		Data:    NewVectorInplace(data),
		Dim:     dim,
		Indices: indices,
	}
	return &t
}

// At get the element at an index, regardless of the dimension
func (t *Tensor) At(idx int) float64 {
	return t.Data.At(idx)
}

// Set set the value to an index, regardless of the dimension
func (t *Tensor) Set(idx int, val interface{}) {
	t.Data.Set(idx, val)
}

// IndexAt get the element at a n-dim index
func (t *Tensor) IndexAt(indices ...int) float64 {
	if len(indices) != len(t.Dim) {
		return 0
	}
	index := 0
	for i, v := range t.Dim {
		index = (index*int(v) + indices[i])
	}
	return t.Data.At(index)
}

// IndexSet set the value to a n-dim index
func (t *Tensor) IndexSet(val interface{}, indices ...int) error {
	if len(indices) != len(t.Dim) {
		return fmt.Errorf("dim unmatched")
	}
	index := 0
	for i, v := range t.Dim {
		index = (index*int(v) + indices[i])
	}
	t.Data.Set(index, val)
	return nil
}

// RowRef return a vector reference to a part of a 2-dim Tensor
func (t *Tensor) RowRef(idx int) *Vector {
	if len(t.Dim) != 2 || idx >= int(t.Dim[0]) {
		return nil
	}
	return t.Data.SubVectorRef(idx*int(t.Dim[1]), int(t.Dim[1]))
}

// Row return a row copy of a 2-dim Tensor
func (t *Tensor) Row(idx int) *Vector {
	if len(t.Dim) != 2 || idx >= int(t.Dim[0]) {
		return nil
	}
	return t.Data.SubVector(idx*int(t.Dim[1]), int(t.Dim[1]))
}

// SetRow set a row with a vector
func (t *Tensor) SetRow(idx int, vec *Vector) error {
	if len(t.Dim) != 2 || idx >= int(t.Dim[0]) || t.Dim[1] != int64(vec.Length) {
		return fmt.Errorf("SETROW FAIL")
	}
	start := int(idx) * int(vec.Length) * int(t.Data.Dtype.Size)
	buffer := bytes.NewBuffer(t.Data.Data[start:])
	buffer.Reset()
	binary.Write(buffer, binary.LittleEndian, vec.Data)
	return nil
}

// InplaceSlice gives a Slice interface to the Tensor data
func (t *Tensor) InplaceSlice() interface{} {
	return t.Data.InplaceSlice()
}

// MakeSlice gives a slice copy of the Tensor data
func (t *Tensor) MakeSlice() interface{} {
	return t.Data.MakeSlice()
}

// VectorToTensor form a 1-dim tensor
func (v *Vector) VectorToTensor() *Tensor {
	var t = Tensor{
		Data: v,
		Dim:  []int64{1, int64(v.Length)},
	}
	return &t
}

// DeserializeTensorPB transforms pb to tensor
func DeserializeTensorPB(pb *proto.Tensor) *Tensor {
	dtype := FlagToDataType[int(pb.Dtype)]
	if int(ProductInt64(pb.Dim))*dtype.Size != len(pb.Content) {
		return nil
	}
	var t = NewTensorInplace(pb.Name, pb.Content, pb.Dim, pb.Indices)
	t.Data.Length /= dtype.Size
	t.Data.Dtype = dtype
	return t
}

// SerializeTensor transforms tensor to pb
func SerializeTensor(t *Tensor) *proto.Tensor {
	var pb = proto.Tensor{
		Name:    t.Name,
		Dim:     t.Dim,
		Indices: t.Indices,
		Content: t.Data.Data,
		Dtype:   proto.TensorDtype(t.Data.Dtype.Flag),
	}
	return &pb
}
