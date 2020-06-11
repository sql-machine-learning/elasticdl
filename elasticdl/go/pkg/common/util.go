// Copyright 2020 The SQLFlow Authors. All rights reserved.
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
	"math"
	"reflect"
)

// CompareFloat compares two float32 number
func CompareFloat(a float64, b float64, tolerance float64) bool {
	diff := math.Abs(a - b)
	mean := math.Abs(a+b) / 2.0
	if math.IsNaN(diff / mean) {
		return true
	}
	return (diff / mean) < tolerance
}

// CompareFloatArray compares two float32/64 array
func CompareFloatArray(a interface{}, b interface{}, tolerance float64) bool {
	vala := reflect.ValueOf(a)
	valb := reflect.ValueOf(b)
	for i := 0; i < vala.Len(); i++ {
		if !CompareFloat(vala.Index(i).Float(), valb.Index(i).Float(), tolerance) {
			return false
		}
	}
	return true
}

// CompareIntArray compares two int8/16/32/64 array
func CompareIntArray(a interface{}, b interface{}) bool {
	vala := reflect.ValueOf(a)
	valb := reflect.ValueOf(b)
	if vala.Len() != valb.Len() {
		return false
	}
	for i := 0; i < vala.Len(); i++ {
		if vala.Index(i).Int() != valb.Index(i).Int() {
			return false
		}
	}
	return true
}
