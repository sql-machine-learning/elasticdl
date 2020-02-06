package common

import (
	"bytes"
	"encoding/binary"
	"math"
	"reflect"
	"unsafe"
)

// Vector definition
type Vector struct {
	Data   []byte
	Length int
	Dtype  Flag
}

// InitializeFunc func
type InitializeFunc func([]byte) error

// ZeroInit return a zero initializer
func ZeroInit() InitializeFunc {
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
func ConstantInit(val interface{}) InitializeFunc {
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
func CopyInit(slice interface{}) InitializeFunc {
	return func(b []byte) error {
		buffer := bytes.NewBuffer(b)
		buffer.Reset()
		return binary.Write(buffer, binary.LittleEndian, slice)
	}
}

// NewEmptyVector create an zero-initialized vector
func NewEmptyVector(length int, flag Flag) *Vector {
	dtype := FlagToDataType[flag]
	bytelen := int(dtype.Size) * length
	var vec = Vector{
		Data:   make([]byte, bytelen, bytelen),
		Dtype:  flag,
		Length: length,
	}
	return &vec
}

// NewInitializedVector create an initialized vector
func NewInitializedVector(length int, flag Flag, initializer InitializeFunc) *Vector {
	dtype := FlagToDataType[flag]
	bytelen := int(dtype.Size) * length
	data := make([]byte, bytelen, bytelen)
	initializer(data)
	var vec = Vector{
		Data:   data,
		Dtype:  flag,
		Length: length,
	}
	return &vec
}

// NewVector create a vector using a slice to initialize
func NewVector(slice interface{}) *Vector {
	v := reflect.ValueOf(slice)
	dtype := TypeToDataType[v.Type()]
	length := v.Len()
	initializer := CopyInit(slice)
	return NewInitializedVector(length, dtype.Flag, initializer)
}

// NewVectorInplace create a vector using a slice to initialize
func NewVectorInplace(slice interface{}) *Vector {
	v := reflect.ValueOf(slice)
	dtype := TypeToDataType[v.Type()]
	length := v.Len()
	bytelen := int(dtype.Size) * length
	sliceHeader := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(v.Pointer())),
		Cap:  int(bytelen),
		Len:  int(bytelen),
	}
	var vec = Vector{
		Data:   *(*[]byte)(unsafe.Pointer(&sliceHeader)),
		Dtype:  dtype.Flag,
		Length: length,
	}
	return &vec
}

// At get the element at an index
func (v *Vector) At(idx int) float64 {
	dtype := FlagToDataType[v.Dtype]
	size := dtype.Size
	switch dtype.Flag {
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
	dtype := FlagToDataType[v.Dtype]
	size := dtype.Size
	byteSet(v.Data, idx, size, val)
}

// InplaceSlice gives a Slice interface to the Vector data
func (v *Vector) InplaceSlice() interface{} {
	dtype := FlagToDataType[v.Dtype]
	sliceHeader := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&v.Data[0])),
		Cap:  int(v.Length),
		Len:  int(v.Length),
	}
	val := reflect.NewAt(dtype.Type, unsafe.Pointer(&sliceHeader)).Elem()
	return val.Interface()
}

// MakeSlice gives a slice copy of the Vector data
func (v *Vector) MakeSlice() interface{} {
	dtype := FlagToDataType[v.Dtype]
	newslice := make([]byte, int(dtype.Size))
	copy(newslice, v.Data)
	sliceHeader := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&newslice[0])),
		Cap:  int(v.Length),
		Len:  int(v.Length),
	}
	val := reflect.NewAt(dtype.Type, unsafe.Pointer(&sliceHeader)).Elem()
	return val.Interface()
}

// SubVectorRef return a reference to a part of the Vector
func (v *Vector) SubVectorRef(begin int, length int) *Vector {
	dtype := FlagToDataType[v.Dtype]
	slice := v.Data[begin*dtype.Size : begin*dtype.Size+length*dtype.Size]
	var vec = Vector{
		Data:   slice,
		Length: length,
		Dtype:  v.Dtype,
	}
	return &vec
}

// SubVector return a part copy of the Vector
func (v *Vector) SubVector(begin int, length int) *Vector {
	dtype := FlagToDataType[v.Dtype]
	slice := v.Data[begin*dtype.Size : begin*dtype.Size+length*dtype.Size]
	newslice := make([]byte, length*dtype.Size, length*dtype.Size)
	copy(newslice, slice)
	var vec = Vector{
		Data:   newslice,
		Length: length,
		Dtype:  v.Dtype,
	}
	return &vec
}
