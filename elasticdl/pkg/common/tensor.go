package common

import "elasticdl.org/elasticdl/pkg/proto"
import "fmt"

type Tensor struct {
	name    string
	value   []float32
	dim     []int64
	indices []int64
}

func DeserializeTensorPB(pb *proto.Tensor, t *Tensor) {
	fmt.Println("hello world")
}
