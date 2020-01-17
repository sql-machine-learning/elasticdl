package common

import (
    "bytes"
	"elasticdl.org/elasticdl/pkg/proto"
	"encoding/binary"
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
	copy(t.Dim, pb.GetDim()
	copy(t.Indices, pb.GetIndices)
	t.Value := make([]float32, len(pb.GetContent())/4)
	br := bytes.NewReader(pb.GetContent())
	binary.Read(br, binary.LittleEndian, &t.Value)
}

// SerializeTensor tensor to pb
func SerializeTensor(t *Tensor, pb *proto.Tensor) {
	pb.Name = t.Name
	pb.Dim = t.Dim
	pb.Indices = t.Indices
	bits := math.Float32bits(t.Value)
	binary.LittleEndian.PutUint32(pb.bytes, bits)
	// set dtype to float32
	pb.Dtype = 6
}
