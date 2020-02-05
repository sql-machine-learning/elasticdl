package common

import (
	"bytes"
	"encoding/binary"
	"math"
	"reflect"
	"unsafe"
	//"fmt"
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
var InvalidDtype = DataType{nil, 0, 0}

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

// TypeToDataType golang reflect.Type -> DataType
var TypeToDataType = make(map[reflect.Type]DataType)

func init() {
	TypeToDataType[Int8Dtype.Type] = Int8Dtype
	TypeToDataType[Int16Dtype.Type] = Int16Dtype
	TypeToDataType[Int32Dtype.Type] = Int32Dtype
	TypeToDataType[Int64Dtype.Type] = Int64Dtype
	TypeToDataType[Float32Dtype.Type] = Float32Dtype
	TypeToDataType[Float64Dtype.Type] = Float64Dtype
}

// Vector definition
type Vector struct {
	Data   []byte
	Length int
	Dtype  DataType
}

// InitialzeFunc func
type InitialzeFunc func([]byte) error

// ZeroInit return a zero initializer
func ZeroInit() InitialzeFunc {
	return func(buffer []byte) error {
		// do nothing since golang has already zero init
		return nil
	}
}

func byteSet(buffer []byte, idx int, size int, val interface{}) {
	switch val.(type) {
	case int8:
		buffer[idx] = byte(val.(int8))
	case int16:
		binary.LittleEndian.PutUint16(buffer[idx*size:idx*size+size], uint16(val.(int16)))
	case int32:
		binary.LittleEndian.PutUint32(buffer[idx*size:idx*size+size], uint32(val.(int32)))
	case int64:
		binary.LittleEndian.PutUint64(buffer[idx*size:idx*size+size], uint64(val.(int64)))
	case float32:
		binary.LittleEndian.PutUint32(buffer[idx*size:idx*size+size], math.Float32bits(val.(float32)))
	case float64:
		binary.LittleEndian.PutUint64(buffer[idx*size:idx*size+size], math.Float64bits(val.(float64)))
	}
}

// ConstantInit return a constant iniitializer
func ConstantInit(val interface{}) InitialzeFunc {
	return func(buffer []byte) error {
		size := int(reflect.ValueOf(val).Type().Size())
		length := len(buffer) / size
		for idx := 0; idx < length; idx++ {
			byteSet(buffer, idx, size, val)
		}
		return nil
	}
}

// CopyInit return a copy initializer
func CopyInit(slice interface{}) InitialzeFunc {
	return func(b []byte) error {
		buffer := bytes.NewBuffer(b)
		buffer.Reset()
		return binary.Write(buffer, binary.LittleEndian, slice)
	}
}

// NewEmptyVector create an zero-initialized vector
func NewEmptyVector(length int, dtype DataType) *Vector {
	bytelen := int(dtype.Size) * length
	var vec = Vector{
		Data:   make([]byte, bytelen, bytelen),
		Dtype:  dtype,
		Length: length,
	}
	return &vec
}

// NewInitializerVector create an initialized vector
func NewInitializerVector(length int, dtype DataType, initializer InitialzeFunc) *Vector {
	bytelen := int(dtype.Size) * length
	data := make([]byte, bytelen, bytelen)
	initializer(data)
	var vec = Vector{
		Data:   data,
		Dtype:  dtype,
		Length: length,
	}
	return &vec
}

// NewVectorFrom create a vector using a slice to initialize
func NewVectorFrom(slice interface{}) *Vector {
	v := reflect.ValueOf(slice)
	dtype := TypeToDataType[v.Type()]
	length := v.Len()
	initializer := CopyInit(slice)
	return NewInitializerVector(length, dtype, initializer)
}

// At get the element at an index
func (v *Vector) At(idx int) float64 {
	size := v.Dtype.Size
	switch v.Dtype.Flag {
	case Int8:
		return float64(int8(v.Data[idx]))
	case Int16:
		return float64(int16(binary.LittleEndian.Uint16(v.Data[idx*size : idx*size+size])))
	case Int32:
		return float64(int32(binary.LittleEndian.Uint32(v.Data[idx*size : idx*size+size])))
	case Int64:
		return float64(int64(binary.LittleEndian.Uint64(v.Data[idx*size : idx*size+size])))
	case Float32:
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(v.Data[idx*size : idx*size+size])))
	case Float64:
		return math.Float64frombits(binary.LittleEndian.Uint64(v.Data[idx*size : idx*size+size]))
	default:
		return 0
	}
}

// Set set the value to an index
func (v *Vector) Set(idx int, val interface{}) {
	size := v.Dtype.Size
	byteSet(v.Data, idx, size, val)
}

// InplaceSlice gives a Slice interface to the Vector data
func (v *Vector) InplaceSlice() interface{} {
	sliceHeader := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&v.Data[0])),
		Cap:  int(v.Length),
		Len:  int(v.Length),
	}
	val := reflect.NewAt(v.Dtype.Type, unsafe.Pointer(&sliceHeader)).Elem()
	return val.Interface()
}

// MakeSlice gives a slice copy of the Vector data
func (v *Vector) MakeSlice() interface{} {
	newslice := make([]byte, int(v.Length*v.Dtype.Size))
	copy(newslice, v.Data)
	sliceHeader := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&newslice[0])),
		Cap:  int(v.Length),
		Len:  int(v.Length),
	}
	val := reflect.NewAt(v.Dtype.Type, unsafe.Pointer(&sliceHeader)).Elem()
	return val.Interface()
}

// SubVectorRef return a reference to a part of the Vector
func (v *Vector) SubVectorRef(begin int, length int) *Vector {
	slice := v.Data[begin*v.Dtype.Size : begin*v.Dtype.Size+length*v.Dtype.Size]
	var vec = Vector{
		Data:   slice,
		Length: length,
		Dtype:  v.Dtype,
	}
	return &vec
}

// SubVector return a part copy of the Vector
func (v *Vector) SubVector(begin int, length int) *Vector {
	slice := v.Data[begin*v.Dtype.Size : begin*v.Dtype.Size+length*v.Dtype.Size]
	newslice := make([]byte, length*v.Dtype.Size, length*v.Dtype.Size)
	copy(newslice, slice)
	var vec = Vector{
		Data:   newslice,
		Length: length,
		Dtype:  v.Dtype,
	}
	return &vec
}
