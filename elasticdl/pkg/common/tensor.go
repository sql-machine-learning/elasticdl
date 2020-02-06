package common

import (
	"bytes"
	"elasticdl.org/elasticdl/pkg/proto"
	"encoding/binary"
	"math"
)

// Tensor defines tensor struct
// TODO(qijun): handle different tensor dtype
type Tensor struct {
	Name    string
	Value   []float32
	Dim     []int64
	Indices []int64
}

// NewTensor create a new n-dim tensor
func NewTensor(dim []int64, name string) *Tensor {
	var t Tensor
	t.Name = name
	t.Value = make([]float32, GetDimProduct(dim))
	t.Dim = dim
	return &t
}

// NewVector create a new 1-dim tensor
func NewVector(dim int64, name string) *Tensor {
	var t Tensor
	t.Value = make([]float32, dim)
	t.Dim = []int64{dim}
	t.Name = name
	return &t
}

// GetDimProduct get the number of the elements of a tensor of this dim
func GetDimProduct(dim []int64) int64 {
	var size int64 = 1
	for _, d := range dim {
		size *= d
	}
	return size
}

// Subtensor get the part reference of the tensor
func (t *Tensor) subTensor(begin int64, len int64) *Tensor {
	if begin+len > GetDimProduct(t.Dim) {
		return nil
	}
	var subt Tensor
	subt.Value = t.Value[begin : begin+len]
	subt.Dim = []int64{len}
	return &subt
}

// AtRow get the row reference of a 2-dim tensor
func (t *Tensor) AtRow(idx int64) *Tensor {
	if len(t.Dim) != 2 || idx >= t.Dim[0] {
		return nil
	}
	begin := t.Dim[1] * idx
	return t.subTensor(begin, t.Dim[1])
}

// DeserializeTensorPB transforms pb to tensor
func DeserializeTensorPB(pb *proto.Tensor) *Tensor {
	var t Tensor
	if pb.Dtype != proto.TensorDtype_DT_FLOAT32 {
		return nil
	}
	if GetDimProduct(pb.Dim)*4 != int64(len(pb.Content)) {
		return nil
	}
	t.Name = pb.Name
	if pb.Dim != nil {
		t.Dim = make([]int64, len(pb.Dim))
		copy(t.Dim, pb.Dim)
	}
	if pb.Indices != nil {
		t.Indices = make([]int64, len(pb.Indices))
		copy(t.Indices, pb.Indices)
	}
	t.Value = make([]float32, len(pb.Content)/4)
	br := bytes.NewReader(pb.GetContent())
	binary.Read(br, binary.LittleEndian, &t.Value)
	return &t
}

// SerializeTensor transforms tensor to pb
func SerializeTensor(t *Tensor) *proto.Tensor {
	var pb proto.Tensor
	pb.Name = t.Name
	if t.Dim != nil {
		pb.Dim = make([]int64, len(t.Dim))
		copy(pb.Dim, t.Dim)
	}
	if t.Indices != nil {
		pb.Indices = make([]int64, len(t.Indices))
		copy(pb.Indices, t.Indices)
	}
	if t.Value != nil {
		pb.Content = make([]byte, GetDimProduct(t.Dim)*4)
		for i, num := range t.Value {
			bits := math.Float32bits(num)
			binary.LittleEndian.PutUint32(pb.Content[(i*4):], bits)
		}
	}
	// set dtype to float32
	pb.Dtype = proto.TensorDtype_DT_FLOAT32
	return &pb
}
