package common

import (
	"math"
	"reflect"
)

// CompareFloat compares two float64 number
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
