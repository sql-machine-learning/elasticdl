package common

import (
	"elasticdl.org/elasticdl/pkg/proto"
	"encoding/binary"
	"fmt"
	"math"
)

// Tensor tensor struct
// TODO(qijun): handle different tensor dtype
type Tensor struct {
	Name    string
	Value   []float32
	Dim     []int64
	Indices []int64
}

// DeserializeTensorPB pb to tensor
func DeserializeTensorPB(pb *proto.Tensor, t *Tensor) {
	t.Name = pb.GetName()
	t.Dim = pb.GetDim()
	t.Indices = pb.GetIndices()
	bits := binary.LittleEndian.Uint32(bytes)
	t.Value = math.Float32frombits(bits)
}

// SerializeTensor tensor to pb
func SerializeTensor(t *Tensor, pb *proto.Tensor) {
	pb.Name = t.Name
	pb.Dim = t.Dim
	pb.Indices = t.Indices
	bits := math.Float32bits(t.value)
	binary.LittleEndian.PutUint32(pb.bytes, bits)
	// set dtype to float32
	pb.Dtype = 6
}
