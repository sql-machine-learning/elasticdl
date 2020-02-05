package common

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestVector(t *testing.T) {
	// test create empty vector
	emptyVec := NewEmptyVector(5, Float32Dtype)
	EvDataExpected := make([]byte, 20)
	assert.Equal(t, EvDataExpected, emptyVec.Data, "empty vector Fail")

	// test create constant init vector
	constinit := ConstantInit(float64(12.5))
	ciVec := NewInitializerVector(5, Float64Dtype, constinit)
	// test inplace slice
	ciVecSlice := ciVec.InplaceSlice().([]float64)
	ciExpected := []float64{12.5, 12.5, 12.5, 12.5, 12.5}
	assert.True(t, CompareFloatArray(ciExpected, ciVecSlice, 0.00001), "constant-init-vector Fail")

	// test At
	assert.Equal(t, 12.5, ciVec.At(0), "Vector.At Fail")

	// test Set
	ciVec.Set(2, float64(31.2))
	assert.Equal(t, 31.2, ciVec.At(2), "Vector.Set Fail")

	i8 := []int8{1, 2, 3}
	i16 := []int16{1, 2, 3}
	i32 := []int32{1, 2, 3}
	i64 := []int64{1, 2, 3}
	f32 := []float32{1, 2, 3}
	f64 := []float64{1, 2, 3}

	// test create vector from slice of different data type
	i8Vec := NewVectorFrom(i8)
	i16Vec := NewVectorFrom(i16)
	i32Vec := NewVectorFrom(i32)
	i64Vec := NewVectorFrom(i64)
	f32Vec := NewVectorFrom(f32)
	f64Vec := NewVectorFrom(f64)

	iExpected := []int{1, 2, 3}
	i8VecSlice := i8Vec.InplaceSlice().([]int8)
	i16VecSlice := i16Vec.InplaceSlice().([]int16)
	i32VecSlice := i32Vec.InplaceSlice().([]int32)
	i64VecSlice := i64Vec.InplaceSlice().([]int64)
	fExpected := []float32{1, 2, 3}
	f32VecSlice := f32Vec.InplaceSlice().([]float32)
	f64VecSlice := f64Vec.InplaceSlice().([]float64)
	assert.True(t, CompareIntArray(iExpected, i8VecSlice), "int8-vector Fail")
	assert.True(t, CompareIntArray(iExpected, i16VecSlice), "int16-vector Fail")
	assert.True(t, CompareIntArray(iExpected, i32VecSlice), "int32-vector Fail")
	assert.True(t, CompareIntArray(iExpected, i64VecSlice), "int64-vector Fail")
	assert.True(t, CompareFloatArray(fExpected, f32VecSlice, 0.00001), "float32-vector Fail")
	assert.True(t, CompareFloatArray(fExpected, f64VecSlice, 0.00001), "float64-vector Fail")

	// test InplaceSlice
	i8VecInplaceSlice := i8Vec.InplaceSlice().([]int8)
	i8VecInplaceSlice[0] = int8(13)
	assert.Equal(t, int8(i8Vec.At(0)), i8VecInplaceSlice[0], "InplaceSlice Fail")

	// test MakeSlice
	i8VecMakeSlice := i8Vec.MakeSlice().([]int8)
	i8VecMakeSlice[0] = int8(23)
	assert.NotEqual(t, int8(i8Vec.At(0)), i8VecMakeSlice[0], "MakeSlice Fail")

	ori := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	oriVec := NewVectorFrom(ori)

	// test SubVectorRef
	oriVecSubRef := oriVec.SubVectorRef(2, 3)
	oriVecSubRefInplaceSlice := oriVecSubRef.InplaceSlice().([]float64)
	subRefExpected := []float64{3, 4, 5}
	assert.True(t, CompareFloatArray(subRefExpected, oriVecSubRefInplaceSlice, 0.00001), "SubVectorRef Fail")
	oriVecSubRef.Set(1, 9.9)
	oriExpected := []float64{1, 2, 3, 9.9, 5, 6, 7, 8}
	oriVecInplcaeSlice := oriVec.InplaceSlice().([]float64)
	assert.True(t, CompareFloatArray(oriVecInplcaeSlice, oriExpected, 0.00001), "SubVectorRef Fail")

	// test SubVectorRef
	oriVecSub := oriVec.SubVector(2, 3)
	oriVecSubInplceSlice := oriVecSub.InplaceSlice().([]float64)
	subExpected := []float64{3, 9.9, 5}
	assert.True(t, CompareFloatArray(subExpected, oriVecSubInplceSlice, 0.00001), "SubVector Fail")
	oriVecSub.Set(1, 99.9)
	oriExpected = []float64{1, 2, 3, 99.9, 5, 6, 7, 8}
	assert.False(t, CompareFloatArray(oriVecInplcaeSlice, oriExpected, 0.00001), "SubVector Fail")
}
