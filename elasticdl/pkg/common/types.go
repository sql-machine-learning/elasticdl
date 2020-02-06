package common

import (
	"reflect"
)

// DataType definition
type DataType struct {
	Type reflect.Type
	Size int
	Flag Flag
}

// Flag DataType.Flag
type Flag = int

// DataType Flag Enum
const (
	Invalid Flag = iota
	Int8
	Int16
	Int32
	Int64
	Float16
	Float32
	Float64
	Bool
)

// InvalidDtype predefined DataType for []int16
var InvalidDtype = DataType{reflect.TypeOf([]byte{0}), 1, 0}

// Int8Dtype predefined DataType for []int16
var Int8Dtype = DataType{reflect.TypeOf([]int8{0}), 1, 1}

// Int16Dtype predefined DataType for []int16
var Int16Dtype = DataType{reflect.TypeOf([]int16{0}), 2, 2}

// Int32Dtype predefined DataType for []int32
var Int32Dtype = DataType{reflect.TypeOf([]int32{0}), 4, 3}

// Int64Dtype predefined DataType for []int64
var Int64Dtype = DataType{reflect.TypeOf([]int64{0}), 8, 4}

// Float16Dtype predefined DataType for []float16
var Float16Dtype = DataType{nil, 0, 5}

// Float32Dtype predefined DataType for []float32
var Float32Dtype = DataType{reflect.TypeOf([]float32{0}), 4, 6}

// Float64Dtype predefined DataType for []float64
var Float64Dtype = DataType{reflect.TypeOf([]float64{0}), 8, 7}

// BoolDtype predefined DataType for Bool
var BoolDtype = DataType{nil, 0, 8}

// TypeToDataType reflect.Type -> DataType
var TypeToDataType = make(map[reflect.Type]*DataType)

// FlagToDataType reflect.Flag -> DataType
var FlagToDataType = make(map[int]*DataType)

func init() {
	TypeToDataType[InvalidDtype.Type] = &InvalidDtype
	TypeToDataType[Int8Dtype.Type] = &Int8Dtype
	TypeToDataType[Int16Dtype.Type] = &Int16Dtype
	TypeToDataType[Int32Dtype.Type] = &Int32Dtype
	TypeToDataType[Int64Dtype.Type] = &Int64Dtype
	TypeToDataType[Float32Dtype.Type] = &Float32Dtype
	TypeToDataType[Float64Dtype.Type] = &Float64Dtype

	FlagToDataType[InvalidDtype.Flag] = &InvalidDtype
	FlagToDataType[Int8Dtype.Flag] = &Int8Dtype
	FlagToDataType[Int16Dtype.Flag] = &Int16Dtype
	FlagToDataType[Int32Dtype.Flag] = &Int32Dtype
	FlagToDataType[Int64Dtype.Flag] = &Int64Dtype
	FlagToDataType[Float32Dtype.Flag] = &Float32Dtype
	FlagToDataType[Float64Dtype.Flag] = &Float64Dtype
}