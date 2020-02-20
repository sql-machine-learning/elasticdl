package commonnew

import (
	"encoding/binary"
	"github.com/stretchr/testify/assert"
	"github.com/tensorflow/tensorflow/tensorflow/go/core/framework/tensor_go_proto"
	"github.com/tensorflow/tensorflow/tensorflow/go/core/framework/tensor_shape_go_proto"
	"math"
	"testing"
)

func TestTensor(t *testing.T) {
	et := NewEmptyTensor([]int64{3, 3}, Float64)
	assert.Equal(t, et.Dims, []int64{3, 3}, "NewEmptyTensor FAIL")

	ev := NewEmptyVector(5, Float64)
	assert.Equal(t, ev.Dims, []int64{5}, "NewEmptyVector FAIL")

	dim := []int64{2, 3}
	slice := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	t1 := NewTensor(slice, dim)

	assert.Equal(t, Slice(t1).([]float32), slice, "NewTensor FAIL")
	assert.Equal(t, t1.Dims, dim, "NewTensor FAIL")

	v1 := NewVector(slice)
	assert.Equal(t, Slice(v1).([]float32), slice, "NewVector FAIL")
	assert.Equal(t, []int64{6}, v1.Dims, "NewVector FAIL")

	r1 := t1.GetRow(1)
	assert.Equal(t, Slice(r1).([]float32), slice[3:6], "GetRow FAIL")
	assert.Equal(t, []int64{3}, r1.Dims, "GetRow FAIL")

	val := NewVector([]float32{30, 40, 50})
	t1.SetRow(1, val)
	assert.Equal(t, Slice(t1).([]float32), []float32{1.0, 2.0, 3.0, 30, 40, 50}, "SetRow FAIL")
}
