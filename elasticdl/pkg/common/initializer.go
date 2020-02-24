package common

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"encoding/binary"
)

// Initializer definition
type Initializer = func(*Tensor) error

func byteSetInt8(buffer []byte, idx int, val interface{}) {
	buffer[idx] = byte(val.(int8))
}

func byteSetInt16(buffer []byte, idx int, val interface{}) {
	binary.LittleEndian.PutUint16(buffer[idx*2:idx*2+2], uint16(val.(int16)))
}

func byteSetInt32(buffer []byte, idx int, val interface{}) {
	binary.LittleEndian.PutUint32(buffer[idx*4:idx*4+4], uint32(val.(int32)))
}

func byteSetInt64(buffer []byte, idx int, val interface{}) {
	binary.LittleEndian.PutUint64(buffer[idx*8:idx*8+8], uint64(val.(int64)))
}

func byteSetFloat32(buffer []byte, idx int, val interface{}) {
	binary.LittleEndian.PutUint32(buffer[idx*4:idx*4+4], math.Float32bits(val.(float32)))
}

func byteSetFloat64(buffer []byte, idx int, val interface{}) {
	binary.LittleEndian.PutUint64(buffer[idx*8:idx*8+8], math.Float64bits(val.(float64)))
}

var byteSetFuncs = map[DataType]func(buffer []byte, idx int, val interface{}){
	Int8: byteSetInt8,
	Int16: byteSetInt16,
	Int32: byteSetInt32,
	Int64: byteSetInt64,
	Float32: byteSetFloat32,
	Float64: byteSetFloat64,
}

// Zero return a zero Initializer
func Zero() Initializer{
	return func(t *Tensor) error {
		for i := range t.Buffer {
			t.Buffer[i] = 0
		}
		return nil
	}
}

// Constant return a constant Initializer
func Constant(n interface{}) Initializer{
	return func(t *Tensor) error {
		if(reflect.TypeOf(n)!=DtypeToValueType[t.Dtype]) {
			return fmt.Errorf("Wrong tensor data type")
		}
		byteSet := byteSetFuncs[t.Dtype]
		length := int(DimProduct(t.Dims))
		for i := 0; i<length; i++ {
			byteSet(t.Buffer, i, n)
		}
		return nil
	}
}

// RandomNorm return a normal Initializer
func RandomNorm(mean float64, std float64) Initializer{
	return func(t *Tensor) error {
		length := int(DimProduct(t.Dims))
		switch t.Dtype {
		case Float32:
			for i := 0; i<length; i++ {
				byteSetFloat32(t.Buffer, i, float32(rand.NormFloat64()*std+mean))
			}
		case Float64:
			for i := 0; i<length; i++ {
				byteSetFloat64(t.Buffer, i, float64(rand.NormFloat64()*std+mean))
			}
		default:
			return fmt.Errorf("Wrong tensor data type")
		}
		return nil
	}
}

// RandomUniform return a uniform Initializer
func RandomUniform(min float64, max float64) Initializer{
	return func(t *Tensor) error {
		length := int(DimProduct(t.Dims))
		factor := max-min
		switch t.Dtype {
		case Float32:
			for i := 0; i<length; i++ {
				byteSetFloat32(t.Buffer, i, rand.Float32()*float32(factor)+float32(min))
			}
		case Float64:
			for i := 0; i<length; i++ {
				byteSetFloat64(t.Buffer, i, rand.Float64()*factor+min)
			}
		default:
			return fmt.Errorf("Wrong tensor data type")
		}
		return nil
	}
}