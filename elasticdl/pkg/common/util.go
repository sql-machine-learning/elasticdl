package common

import "math"

// CompareFloat compares two float32 number
func CompareFloat(a float32, b float32, tolerance float64) bool {
	x := float64(a)
	y := float64(b)
	diff := math.Abs(x - y)
	mean := math.Abs(x+y) / 2.0
	if math.IsNaN(diff / mean) {
		return true
	}
	return (diff / mean) < tolerance
}

// CompareFloatArray compares two float32 array
func CompareFloatArray(a []float32, b []float32, tolerance float64) bool {
	for i, num := range a {
		if !CompareFloat(num, b[i], tolerance) {
			return false
		}
	}
	return true
}
