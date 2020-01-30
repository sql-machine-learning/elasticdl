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

// NewTensor create a new tensor with name and dim
func NewTensor(name string, dim []int64) *Tensor {
	var t Tensor
	t.Name = name
	t.Value = make([]float32, GetDimProduct(dim))
	t.Dim = dim
	return &t
}

// NewEmbeddingVector create a new tensor with dim
func NewEmbeddingVector(dim int64) *Tensor {
	var t Tensor
	t.Value = make([]float32, dim)
	t.Dim = []int64{1, dim}
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
func (t *Tensor) Subtensor(begin int64, len int64) *Tensor {
	if begin+len > GetDimProduct(t.Dim) {
		return nil
	}
	var subt Tensor
	subt.Value = t.Value[begin : begin+len]
	subt.Dim = []int64{1, len}
	return &subt
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
	t.Dim = make([]int64, len(pb.Dim))
	copy(t.Dim, pb.Dim)
	t.Indices = make([]int64, len(pb.Indices))
	copy(t.Indices, pb.Indices)
	t.Value = make([]float32, len(pb.Content)/4)
	br := bytes.NewReader(pb.GetContent())
	binary.Read(br, binary.LittleEndian, &t.Value)
	return &t
}

// SerializeTensor transforms tensor to pb
func SerializeTensor(t *Tensor) *proto.Tensor {
	var pb proto.Tensor
	pb.Name = t.Name
	pb.Dim = make([]int64, len(t.Dim))
	copy(pb.Dim, t.Dim)
	pb.Indices = make([]int64, len(t.Indices))
	copy(pb.Indices, t.Indices)
	pb.Content = make([]byte, GetDimProduct(t.Dim)*4)
	for i, num := range t.Value {
		bits := math.Float32bits(num)
		binary.LittleEndian.PutUint32(pb.Content[(i*4):], bits)
	}
	// set dtype to float32
	pb.Dtype = proto.TensorDtype_DT_FLOAT32
	return &pb
}
