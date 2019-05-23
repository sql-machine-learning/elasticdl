//go:generate protoc --go_out=plugins=grpc:. -I ../../swamp/coordinator/proto ../../swamp/coordinator/proto/service.proto

package coordinator

import (
	context "context"
	"errors"
	"fmt"
	"math/rand"

	"github.com/golang/protobuf/proto"
	empty "github.com/golang/protobuf/ptypes/empty"
)

type server struct {
	spec *HyperParameterSearchSpec
}

func sanityCheck(spec *HyperParameterSearchSpec) error {
	if spec == nil {
		return errors.New("Nil HyperParameterSearchSpec")
	}

	params := spec.GetParam()
	if len(params) == 0 {
		return errors.New("Empty HyperParameterSearchSpec")
	}

	for k, s := range params {
		switch st := s.Search.(type) {
		case *HyperParameterSearch_SampleSize:
			if st.SampleSize <= 0 {
				return errors.New("Illegal random sample size for parameter: " + k)
			}
		case *HyperParameterSearch_LinearStep:
			return errors.New("Linear search type not supported yet for parameter: " + k)
		case *HyperParameterSearch_LogStep:
			return errors.New("Log search type not supported yet for parameter: " + k)
		case nil:
			return errors.New("Search type unspecified for parameter: " + k)
		default:
			panic("Unknown search type for parameter: " + k)
		}

		switch rt := s.GetRange().Range.(type) {
		case *HyperParameterRange_IntRange:
			if rt.IntRange.Min >= rt.IntRange.Max {
				return errors.New("Illegal search range for parameter: " + k)
			}
		case *HyperParameterRange_DoubleRange:
			if rt.DoubleRange.Min >= rt.DoubleRange.Max {
				return errors.New("Illegal search range for parameter: " + k)
			}
		case *HyperParameterRange_CategoricalRange:
			if len(rt.CategoricalRange.GetValue()) == 0 {
				return errors.New("Empty search range for parameter: " + k)
			}
		case nil:
			return errors.New("Search range unspecified for parameter: " + k)
		default:
			panic("Unknown search range for parameter: " + k)
		}
	}

	return nil
}

// NewServer creates and returns a server with given search spec
func NewServer(spec *HyperParameterSearchSpec) (*server, error) {
	if err := sanityCheck(spec); err != nil {
		return nil, err
	}

	return &server{
		// Clone the spec for safety.
		spec: proto.Clone(spec).(*HyperParameterSearchSpec),
	}, nil
}

func newSample(s *HyperParameterRange) *HyperParameterValue {
	r := &HyperParameterValue{}
	switch st := s.Range.(type) {
	case *HyperParameterRange_IntRange:
		r.Value = &HyperParameterValue_IntValue{st.IntRange.Min + rand.Int31n(st.IntRange.Max-st.IntRange.Min+1)}
	case *HyperParameterRange_DoubleRange:
		r.Value = &HyperParameterValue_DoubleValue{st.DoubleRange.Min + rand.Float64()*(st.DoubleRange.Max-st.DoubleRange.Min)}
	case *HyperParameterRange_CategoricalRange:
		r.Value = &HyperParameterValue_CategoricalValue{st.CategoricalRange.Value[rand.Intn(len(st.CategoricalRange.Value))]}
	}

	return r
}

// Implements CoordinatorServer interface
func (s *server) GetTask(context.Context, *empty.Empty) (*Task, error) {
	// TODO: create task intelligently, for now, just create a random search training task.
	hps := &HyperParameters{Param: map[string]*HyperParameterValue{}}
	for k, s := range s.spec.GetParam() {
		// TODO: implement algorithm to use sample sizes.
		hps.Param[k] = newSample(s.Range)
	}
	// TODO: keep track of task ids
	return &Task{
		TaskId: 0,
		Task: &Task_TrainingTask{
			// TODO: populate base
			&TrainingTask{HyperParameters: hps},
		},
	}, nil
}

func (s *server) PushResult(_ context.Context, r *TaskResult) (*empty.Empty, error) {
	fmt.Println(r)
	return &empty.Empty{}, nil
}
