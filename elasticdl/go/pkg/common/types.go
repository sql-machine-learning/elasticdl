// Copyright 2020 The ElasticDL Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package common

import (
	"github.com/tensorflow/tensorflow/tensorflow/go/core/framework/types_go_proto"
	"reflect"
)

// DataType alias
type DataType = types_go_proto.DataType

// const alias for dtype
const (
	Invalid = types_go_proto.DataType_DT_INVALID
	Int8    = types_go_proto.DataType_DT_INT8
	Int16   = types_go_proto.DataType_DT_INT16
	Int32   = types_go_proto.DataType_DT_INT32
	Int64   = types_go_proto.DataType_DT_INT64
	Float16 = types_go_proto.DataType_DT_BFLOAT16
	Float32 = types_go_proto.DataType_DT_FLOAT
	Float64 = types_go_proto.DataType_DT_DOUBLE
	Bool    = types_go_proto.DataType_DT_BOOL
)

// DtypeSize Dtype -> size
var DtypeSize = make(map[types_go_proto.DataType]int32)

// DtypeToSliceType Dtype -> reflect.Type
var DtypeToSliceType = make(map[types_go_proto.DataType]reflect.Type)

// DtypeToValueType Dtype -> reflect.Type
var DtypeToValueType = make(map[types_go_proto.DataType]reflect.Type)

// SliceTypeToDtype reflect.Type -> Dtype
var SliceTypeToDtype = make(map[reflect.Type]types_go_proto.DataType)

func init() {
	DtypeSize[types_go_proto.DataType_DT_INVALID] = 1
	DtypeSize[types_go_proto.DataType_DT_INT8] = 1
	DtypeSize[types_go_proto.DataType_DT_INT16] = 2
	DtypeSize[types_go_proto.DataType_DT_INT32] = 4
	DtypeSize[types_go_proto.DataType_DT_INT64] = 8
	DtypeSize[types_go_proto.DataType_DT_BFLOAT16] = 2
	DtypeSize[types_go_proto.DataType_DT_FLOAT] = 4
	DtypeSize[types_go_proto.DataType_DT_DOUBLE] = 8
	DtypeSize[types_go_proto.DataType_DT_BOOL] = 1

	DtypeToSliceType[types_go_proto.DataType_DT_INVALID] = reflect.TypeOf([]byte{0})
	DtypeToSliceType[types_go_proto.DataType_DT_INT8] = reflect.TypeOf([]int8{0})
	DtypeToSliceType[types_go_proto.DataType_DT_INT16] = reflect.TypeOf([]int16{0})
	DtypeToSliceType[types_go_proto.DataType_DT_INT32] = reflect.TypeOf([]int32{0})
	DtypeToSliceType[types_go_proto.DataType_DT_INT64] = reflect.TypeOf([]int64{0})
	//DtypeToSliceType[types_go_proto.DataType_DT_FLOAT16] = reflect.TypeOf([]float16{0})
	DtypeToSliceType[types_go_proto.DataType_DT_FLOAT] = reflect.TypeOf([]float32{0})
	DtypeToSliceType[types_go_proto.DataType_DT_DOUBLE] = reflect.TypeOf([]float64{0})
	DtypeToSliceType[types_go_proto.DataType_DT_BOOL] = reflect.TypeOf([]bool{true})

	DtypeToValueType[types_go_proto.DataType_DT_INVALID] = reflect.TypeOf(byte(0))
	DtypeToValueType[types_go_proto.DataType_DT_INT8] = reflect.TypeOf(int8(0))
	DtypeToValueType[types_go_proto.DataType_DT_INT16] = reflect.TypeOf(int16(0))
	DtypeToValueType[types_go_proto.DataType_DT_INT32] = reflect.TypeOf(int32(0))
	DtypeToValueType[types_go_proto.DataType_DT_INT64] = reflect.TypeOf(int64(0))
	//DtypeToValueType[types_go_proto.DataType_DT_FLOAT16] = reflect.TypeOf(float16(0))
	DtypeToValueType[types_go_proto.DataType_DT_FLOAT] = reflect.TypeOf(float32(0))
	DtypeToValueType[types_go_proto.DataType_DT_DOUBLE] = reflect.TypeOf(float64(0))
	DtypeToValueType[types_go_proto.DataType_DT_BOOL] = reflect.TypeOf(bool(true))

	SliceTypeToDtype[reflect.TypeOf([]byte{0})] = types_go_proto.DataType_DT_INVALID
	SliceTypeToDtype[reflect.TypeOf([]int8{0})] = types_go_proto.DataType_DT_INT8
	SliceTypeToDtype[reflect.TypeOf([]int16{0})] = types_go_proto.DataType_DT_INT16
	SliceTypeToDtype[reflect.TypeOf([]int32{0})] = types_go_proto.DataType_DT_INT32
	SliceTypeToDtype[reflect.TypeOf([]int64{0})] = types_go_proto.DataType_DT_INT64
	//SliceTypeToDtype[reflect.TypeOf([]float16{0})] = types_go_proto.DataType_DT_FLOAT16
	SliceTypeToDtype[reflect.TypeOf([]float32{0})] = types_go_proto.DataType_DT_FLOAT
	SliceTypeToDtype[reflect.TypeOf([]float64{0})] = types_go_proto.DataType_DT_DOUBLE
	SliceTypeToDtype[reflect.TypeOf([]bool{true})] = types_go_proto.DataType_DT_BOOL
}
