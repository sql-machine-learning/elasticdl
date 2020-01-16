package common

import "elasticdl/proto"

type Tensor struct {
	name    string
	value   []float32
	dim     []int64
	indices []int64
}


func deserializeTensorPB(proto.Tensor* pb, Tensor* tensor) {
}