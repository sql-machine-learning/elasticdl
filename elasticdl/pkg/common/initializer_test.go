package common

import (
	"fmt"
	"testing"
)

func TestInitializer(t *testing.T) {
	tensor := NewEmptyTensor([]int64{10}, Float32)

	constinit := Constant(float32(3.14))
	constinit(tensor)
	fmt.Println(Slice(tensor).([]float32))

	zeroinit := Zero()
	zeroinit(tensor)
	fmt.Println(Slice(tensor).([]float32))

	norminit := RandomNorm(0, 1.0, 0)
	norminit(tensor)
	fmt.Println(Slice(tensor).([]float32))

	uniforminit := RandomUniform(-1.0, 1.0, 0)
	uniforminit(tensor)
	fmt.Println(Slice(tensor).([]float32))

	truncatenorminit := TruncatedNormal(-1.0, 1.0, 0)
	truncatenorminit(tensor)
	fmt.Println(Slice(tensor).([]float32))
}
