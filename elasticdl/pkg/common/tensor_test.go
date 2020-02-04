package common

import (
	"elasticdl.org/elasticdl/pkg/proto"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestTensorInit(t *testing.T) {
	i1 := []int64{1, 3, 5, 8, 9}
	d1 := []int64{2, 3}
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	t1 := NewTensor("t1", v1, d1, i1)

	assert.Equal(t, t1.Name, "t1")
	assert.Equal(t, *t1.InplaceSlice().(*[]float32), v1)
	assert.Equal(t, []int64(t1.Dim), d1)
	assert.Equal(t, []int64(t1.Indices), i1)
}

func TestTensorSerialization(t *testing.T) {
	i1 := []int64{1, 3, 5, 8, 9}
	d1 := []int64{2, 3}
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	t1 := NewTensor("t1", v1, d1, i1)

	var p *proto.Tensor = SerializeTensor(t1)
	assert.NotNil(t, p)

	var t2 *Tensor = DeserializeTensorPB(p)

	assert.Equal(t, *t1.InplaceSlice().(*[]float32), *t2.InplaceSlice().(*[]float32))
}
