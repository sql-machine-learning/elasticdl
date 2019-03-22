package coordinator

import (
	fmt "fmt"
	"testing"
)

func newServerFail(spec *HyperParameterSearchSpec, t *testing.T) {
	s, err := NewServer(spec)
	if s != nil {
		t.Error("Expected server to be nil")
	}

	if err == nil {
		t.Error("Expected err ro be non-nil")
	}
	fmt.Println(err)
}

func newServerSuccess(spec *HyperParameterSearchSpec, t *testing.T) {
	s, err := NewServer(spec)
	if s == nil {
		t.Error("Expected server to be non-nil")
	}

	if err != nil {
		t.Error("Expected err ro be nil")
	}
}
func TestCreateServer(t *testing.T) {
	newServerFail(nil, t)

	spec := &HyperParameterSearchSpec{}
	newServerFail(spec, t)

	spec.Param = make(map[string]*HyperParameterSearch)
	spec.Param["succ"] = &HyperParameterSearch{
		Range: &HyperParameterRange{
			Range: &HyperParameterRange_DoubleRange{&DoubleRange{Min: 1.0, Max: 100.0}},
		},
		Search: &HyperParameterSearch_SampleSize{5},
	}
	// Random search supported.
	newServerSuccess(spec, t)

	// Linear search unsupported.
	spec.Param["fail"] = &HyperParameterSearch{
		Range: &HyperParameterRange{
			Range: &HyperParameterRange_DoubleRange{&DoubleRange{Min: 1.0, Max: 100.0}},
		},
		Search: &HyperParameterSearch_LinearStep{5.0},
	}
	newServerFail(spec, t)

	// Log search unsupported.
	spec.Param["fail"] = &HyperParameterSearch{
		Range: &HyperParameterRange{
			Range: &HyperParameterRange_DoubleRange{&DoubleRange{Min: 1.0, Max: 100.0}},
		},
		Search: &HyperParameterSearch_LogStep{5.0},
	}
	newServerFail(spec, t)

	// Unspecified search method.
	spec.Param["fail"] = &HyperParameterSearch{
		Range: &HyperParameterRange{
			Range: &HyperParameterRange_DoubleRange{&DoubleRange{Min: 1.0, Max: 100.0}},
		},
	}
	newServerFail(spec, t)

	// Wrong double range
	spec.Param["fail"] = &HyperParameterSearch{
		Range: &HyperParameterRange{
			Range: &HyperParameterRange_DoubleRange{&DoubleRange{Min: 100.0, Max: 1.0}},
		},
		Search: &HyperParameterSearch_SampleSize{5},
	}
	newServerFail(spec, t)

	// Wrong int range
	spec.Param["fail"] = &HyperParameterSearch{
		Range: &HyperParameterRange{
			Range: &HyperParameterRange_IntRange{&IntRange{Min: 100, Max: 1}},
		},
		Search: &HyperParameterSearch_SampleSize{5},
	}
	newServerFail(spec, t)

	// empty categorical range
	spec.Param["fail"] = &HyperParameterSearch{
		Range: &HyperParameterRange{
			Range: &HyperParameterRange_CategoricalRange{},
		},
		Search: &HyperParameterSearch_SampleSize{5},
	}
	newServerFail(spec, t)
}

func TestRpcMethods(t *testing.T) {
	// Creates server with supported searches.
	spec := &HyperParameterSearchSpec{}
	spec.Param = make(map[string]*HyperParameterSearch)
	spec.Param["p1"] = &HyperParameterSearch{
		Range: &HyperParameterRange{
			Range: &HyperParameterRange_DoubleRange{&DoubleRange{Min: 1.0, Max: 100.0}},
		},
		Search: &HyperParameterSearch_SampleSize{5},
	}
	spec.Param["p2"] = &HyperParameterSearch{
		Range: &HyperParameterRange{
			Range: &HyperParameterRange_IntRange{&IntRange{Min: 1, Max: 100}},
		},
		Search: &HyperParameterSearch_SampleSize{5},
	}
	spec.Param["p3"] = &HyperParameterSearch{
		Range: &HyperParameterRange{
			Range: &HyperParameterRange_CategoricalRange{
				&CategoricalRange{Value: []string{"v1", "v2", "v3"}},
			}},
		Search: &HyperParameterSearch_SampleSize{5},
	}

	s, _ := NewServer(spec)

	for i := 0; i < 10; i++ {
		r, _ := s.GetTask(nil, nil)
		fmt.Println(r)

		p := r.GetTrainingTask().GetHyperParameters().GetParam()
		dv := p["p1"].GetDoubleValue()
		if dv < 1.0 || dv > 100.0 {
			t.Error("rand double value failed")
		}
		iv := p["p2"].GetIntValue()
		if iv < 1 || iv > 100 {
			t.Error("rand int value failed")
		}
		cv := p["p3"].GetCategoricalValue()
		if cv != "v1" && cv != "v2" && cv != "v3" {
			t.Error("rand categorical value failed")
		}
	}
}
