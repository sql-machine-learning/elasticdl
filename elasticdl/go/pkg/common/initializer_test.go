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
