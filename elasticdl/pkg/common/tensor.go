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

func GetTensorSize(t* Tensor) int64 {
    var size int64 = 1
    for _, d := range t.Dim {
        size *= d
    }
    return size
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
	pb.Content = make([]bytes, GetTensorSize(t) * 4)
	for i, num := range t.Value {
	    bits := math.Float64bits(num)
	    binary.LittleEndian.PutUint64(pb.Content[i:], bits)
	}
	// set dtype to float32
	pb.Dtype = 6
}
