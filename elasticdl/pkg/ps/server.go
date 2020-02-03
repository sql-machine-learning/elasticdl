package ps

import (
	"context"
	"elasticdl.org/elasticdl/pkg/common"
	pb "elasticdl.org/elasticdl/pkg/proto"
	"fmt"
	empty "github.com/golang/protobuf/ptypes/empty"
	"google.golang.org/grpc"
	"log"
	"net"
	"sync"
)

type server struct {
	pb.PserverServer
	Param *Parameter
	Opt   Optimizer
	PSID  int32
	lock  *sync.Mutex
}

func newServer(PSID int32, opt string, lr float32) *server {
	var ps server
	ps.Param = NewParameter()
	if opt == "SGD" {
		ps.Opt = &NewSGDOptimizer(lr)
	}
	ps.PSID = PSID
	ps.lock = &sync.Mutex{}
	return &ps
}

func (s *server) PullVariable(ctx context.Context, in *pb.PullVariableRequest) (*pb.PullVariableResponse, error) {
	var res pb.PullVariableResponse
	if !s.Param.InitStatus {
		res.ModelInitStatus = false
		return &res, nil
	}
	res.Model.Version = s.Param.Version
	if s.Param.Version > in.CurrentModelVersion {
		for _, v := range s.Param.NonEmbeddingParam {
			res.Model.Param = append(res.Model.Param, common.SerializeTensor(v))
		}
	}
	s.Param.InitStatus = true
	return &res, nil
}

func (s *server) PullEmbeddingVector(ctx context.Context, in *pb.PullEmbeddingVectorRequest) (*pb.Tensor, error) {
	if in.Ids == nil {
		return &pb.Tensor{}, nil
	}
	table := s.Param.GetEmbeddingParam(in.Name)
	if table == nil {
		return &pb.Tensor{}, fmt.Errorf("Request embedding Table %s not found in Param", in.Name)
	}
	vectors := table.GetEmbeddingVectors(in.Ids)
	newDim := []int64{int64(len(in.Ids))}
	newDim = append(newDim, table.Dim)
	t := common.Tensor{in.Name, vectors, newDim, in.Ids}
	return common.SerializeTensor(&t), nil
}

func (s *server) PushModel(ctx context.Context, in *pb.Model) (*empty.Empty, error) {
	s.lock.Lock()
	var err error
	s.Param, err = DeserializeModelPB(in)
	s.lock.Unlock()
	return &empty.Empty{}, err
}

func (s *server) PushEmbeddingInfo(ctx context.Context, in *pb.Model) (*empty.Empty, error) {
	s.lock.Lock()
	var err error
	s.Param, err = DeserializeModelPB(in)
	s.lock.Unlock()
	return &empty.Empty{}, err
}

func (s *server) PushGradient(ctx context.Context, in *pb.PushGradientRequest) (*pb.PushGradientResponse, error) {
	// TODO(qijun) only support async now
	var res pb.PushGradientResponse
	var grads []*common.Tensor
	for _, gradPB := range in.Gradients {
		grad := common.DeserializeTensorPB(gradPB)
		grads = append(grads, grad)
	}
	err := s.Opt.ApplyGradients(grads, s.Param)
	res.Accepted = true
	res.ModelVersion = s.Param.Version
	return &res, err
}

// CreateServer creates a PS server and starts the serving. Set serverDone when finishes.
func CreateServer(address string, PSID int32, opt string, lr float32, serverDone chan bool) {
	lis, err := net.Listen("tcp", address)
	if err != nil {
		log.Fatalf("failed to start PS: %v", err)
	}
	// TODO: set maxReceiveMessageSize (default is 4M, too small for elasticdl), maxConcurrentStreams
	grpcServer := grpc.NewServer()
	s := newServer(PSID, opt, lr)
	pb.RegisterPserverServer(grpcServer, s)
	go startServe(grpcServer, lis, serverDone)
}

func startServe(server *grpc.Server, lis net.Listener, serverDone chan bool) {
	err := server.Serve(lis)
	if err != nil {
		log.Fatalf("GRPC failed to serve: %v", err)
	}
	serverDone <- true
}
