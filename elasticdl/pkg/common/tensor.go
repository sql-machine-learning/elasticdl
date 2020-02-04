package common

// #cgo LDFLAGS: -L../c -lvector -lm
// #include "../c/vector.h"
import "C"
import (
	"elasticdl.org/elasticdl/pkg/proto"
	"fmt"
	"reflect"
	"runtime"
	"unsafe"
)

// DataType definition
type DataType struct {
	Type reflect.Type
	Size int64
	Flag int64
}

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

// FlagToDataType Pb dtype -> DataType
var FlagToDataType = make(map[int64]DataType)

func init() {
	TypeToDataType[Int16Dtype.Type] = Int16Dtype
	TypeToDataType[Int32Dtype.Type] = Int32Dtype
	TypeToDataType[Int64Dtype.Type] = Int64Dtype
	TypeToDataType[Float32Dtype.Type] = Float32Dtype
	TypeToDataType[Float64Dtype.Type] = Float64Dtype
	FlagToDataType[Int16Dtype.Flag] = Int16Dtype
	FlagToDataType[Int32Dtype.Flag] = Int32Dtype
	FlagToDataType[Int64Dtype.Flag] = Int64Dtype
	FlagToDataType[Float32Dtype.Flag] = Float32Dtype
	FlagToDataType[Float64Dtype.Flag] = Float64Dtype
}

// Vector definition
type Vector = C.Vector

// InitialzeFunc func
type InitialzeFunc func(unsafe.Pointer)

// ZeroInit return a closure
func ZeroInit(length int, dtype DataType) InitialzeFunc {
	return func(ptr unsafe.Pointer) {
		C.memset(ptr, 0, C.ulong(length*int(dtype.Size)))
	}
}

// NewEmptyVector create an zero-initialized vector
func NewEmptyVector(length int, dtype DataType, initializer InitialzeFunc) *Vector {
	var vec = new(Vector)

	vec.Dtype = (C.ulong)(dtype.Flag)
	vec.Length = (C.ulong)(length)
	vec.ByteLength = (C.ulong)(int(dtype.Size) * length)
	vec.Data = C.malloc(vec.ByteLength)
	if initializer == nil {
		initializer = ZeroInit(length, dtype)
	}
	initializer(vec.Data)
	runtime.SetFinalizer(vec, func(temp *Vector) {
		C.free(temp.Data)
	})
	return vec
}

// NewVector create a vector using a slice to initialize
func NewVector(slice interface{}) *Vector {
	v := reflect.ValueOf(slice)
	dtype := TypeToDataType[v.Type()]
	length := v.Len()
	vec := NewEmptyVector(length, dtype, nil)
	C.memcpy(vec.Data, unsafe.Pointer(v.Pointer()), vec.ByteLength)
	return vec
}

// At get the element at an index
func (v *Vector) At(idx int) float64 {
	return float64(C.IndexAtVector(v, C.ulong(idx)))
}

// Set set the value to an index
func (v *Vector) Set(idx int, val float64) {
	C.SetVector(v, C.ulong(idx), C.double(val))
}

// InplaceSlice gives a Slice interface to the Vector data
func (v *Vector) InplaceSlice() interface{} {
	sliceHeader := reflect.SliceHeader{
		Data: uintptr(v.Data),
		Cap:  int(v.Length),
		Len:  int(v.Length),
	}
	val := reflect.NewAt(FlagToDataType[int64(v.Dtype)].Type, unsafe.Pointer(&sliceHeader))
	return val.Interface()
}

// MakeSlice gives a slice copy of the Vector data
func (v *Vector) MakeSlice() interface{} {
	val := reflect.MakeSlice(FlagToDataType[int64(v.Dtype)].Type, int(v.Length), int(v.Length))
	C.memcpy(unsafe.Pointer(val.Pointer()), v.Data, v.ByteLength)
	return val.Interface()
}

// Dim definition
type Dim []int64

// Product of elements in Dim, indicate the number of elements
func (d Dim) Product() int64 {
	var size int64 = 1
	for _, v := range d {
		size *= v
	}
	return size
}

// Len of the Dim array
func (d Dim) Len() int {
	return len(d)
}

// Index of the Dim array
func (d Dim) Index(idx int) int64 {
	return d[idx]
}

// Indices definition
type Indices []int64

// Len of the Indices array
func (i Indices) Len() int {
	return len(i)
}

// Tensor definition
type Tensor struct {
	Name    string
	Data    *Vector
	Dim     Dim
	Indices Indices
	Dtype   DataType
}

// NewEmptyTensor create an uninitialized tensor
func NewEmptyTensor(name string, dtype DataType, dim ...int64) *Tensor {
	var t = new(Tensor)
	t.Name = name
	t.Dim = dim
	t.Data = NewEmptyVector(int(t.Dim.Product()), dtype, nil)
	t.Dtype = dtype
	return t
}

// NewTensor create a tensor using giving slice and dim, and name
func NewTensor(name string, data interface{}, dim Dim, indices Indices) *Tensor {
	var t = new(Tensor)
	t.Dim = dim
	t.Indices = indices
	t.Data = NewVector(data)
	t.Name = name
	t.Dtype = FlagToDataType[int64(t.Data.Dtype)]
	return t
}

// FlatAt get the element at an index, regardless of the dimension
func (t *Tensor) FlatAt(idx int) float64 {
	return t.Data.At(idx)
}

// FlatSet set the value to an index, regardless of the dimension
func (t *Tensor) FlatSet(idx int, val float64) {
	t.Data.Set(idx, val)
}

// At get the element at a n-dim index
func (t *Tensor) At(indices ...int) float64 {
	if len(indices) != t.Dim.Len() {
		return 0
	}
	index := 0
	for i, v := range t.Dim {
		index = (index*int(v) + indices[i])
	}
	return t.Data.At(index)
}

// Set set the value to a n-dim index
func (t *Tensor) Set(val float64, indices ...int) error {
	if len(indices) != t.Dim.Len() {
		return fmt.Errorf("UNMATCH DIM")
	}
	index := 0
	for i, v := range t.Dim {
		index = (index*int(v) + indices[i])
	}
	t.Data.Set(index, val)
	return nil
}

// SubVectorRef return a reference to a part of the Vector
func (v *Vector) SubVectorRef(begin int, length int) *Vector {
	var vec = new(Vector)
	vec.Data = C.AddressAt(v, C.ulong(begin))
	vec.Dtype = v.Dtype
	vec.Length = (C.ulong)(length)
	vec.ByteLength = (C.ulong)(int64(length) * FlagToDataType[int64(vec.Dtype)].Size)
	return vec
}

// RowRef return a reference to a part of a 2-dim Tensor
func (t *Tensor) RowRef(idx int) *Vector {
	if t.Dim.Len() != 2 || idx >= int(t.Dim[0]) {
		return nil
	}
	return t.Data.SubVectorRef(idx*int(t.Dim[1]), int(t.Dim[1]))
}

// SubVector return a part copy of the Vector
func (v *Vector) SubVector(begin int, length int) *Vector {
	dtype := FlagToDataType[int64(v.Dtype)]
	var vec = NewEmptyVector(length, dtype, nil)
	C.memcpy(vec.Data, C.AddressAt(v, C.ulong(begin)), vec.ByteLength)
	return vec
}

// Row return a row copy of a 2-dim Tensor
func (t *Tensor) Row(idx int) *Vector {
	if t.Dim.Len() != 2 || idx >= int(t.Dim[0]) {
		return nil
	}
	return t.Data.SubVector(idx*int(t.Dim[1]), int(t.Dim[1]))
}

// SetRow set a row with a vector
func (t *Tensor) SetRow(idx int, vec *Vector) error {
	if t.Dim.Len() != 2 || idx >= int(t.Dim[0]) || t.Dim[1] != int64(vec.Length) {
		return fmt.Errorf("SETROW FAIL")
	}
	start := int(idx) * int(vec.Length) * int(t.Dtype.Size)
	C.memcpy(unsafe.Pointer(uintptr(t.Data.Data)+uintptr(start)), vec.Data, vec.ByteLength)
	return nil
}

// InplaceSlice gives a Slice interface to the Tensor data
func (t *Tensor) InplaceSlice() interface{} {
	sliceHeader := reflect.SliceHeader{
		Data: uintptr(t.Data.Data),
		Cap:  int(t.Data.Length),
		Len:  int(t.Data.Length),
	}
	val := reflect.NewAt(t.Dtype.Type, unsafe.Pointer(&sliceHeader))
	return val.Interface()
}

// MakeSlice gives a slice copy of the Tensor data
func (t *Tensor) MakeSlice() interface{} {
	val := reflect.MakeSlice(t.Dtype.Type, int(t.Data.Length), int(t.Data.Length))
	C.memcpy(unsafe.Pointer(val.Pointer()), t.Data.Data, t.Data.ByteLength)
	return val.Interface()
}

// DeserializeTensorPB transforms pb to tensor
func DeserializeTensorPB(pb *proto.Tensor) *Tensor {
	if Dim(pb.Dim).Product()*FlagToDataType[int64(pb.Dtype)].Size != int64(len(pb.Content)) {
		return nil
	}
	var t = NewEmptyTensor(pb.Name, FlagToDataType[int64(pb.Dtype)], pb.Dim...)
	t.Indices = make([]int64, len(pb.Indices))
	copy(t.Indices, pb.Indices)
	C.memcpy(t.Data.Data, unsafe.Pointer(&pb.Content[0]), C.ulong(len(pb.Content)))
	return t
}

// SerializeTensor transforms tensor to pb
func SerializeTensor(t *Tensor) *proto.Tensor {
	var pb proto.Tensor
	pb.Name = t.Name
	pb.Dim = make([]int64, len(t.Dim))
	copy(pb.Dim, t.Dim)
	pb.Indices = make([]int64, len(t.Indices))
	copy(pb.Indices, t.Indices)
	pb.Content = make([]byte, int(t.Data.ByteLength))
	C.memcpy(unsafe.Pointer(&pb.Content[0]), t.Data.Data, t.Data.ByteLength)
	pb.Dtype = proto.TensorDtype(t.Data.Dtype)
	return &pb
}
