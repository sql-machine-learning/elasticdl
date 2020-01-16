package common

type Tensor struct {
	name    string
	value   []float32
	dim     []int64
	indices []int64
}
