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

// Server defines servicer of ps
type Server struct {
	pb.PserverServer
	Param *Parameter
	Opt   Optimizer
	PSID  int32
	lock  *sync.Mutex
}

// NewServer creates a Server instance
func NewServer(PSID int32, opt string, lr float32) *Server {
	var ps Server
	ps.Param = NewParameter()
	if opt == "SGD" {
		ps.Opt = NewSGDOptimizer(lr)
	}
	ps.PSID = PSID
	ps.lock = &sync.Mutex{}
	return &ps
}

// PullVariable pulls variable from server
func (s *Server) PullVariable(ctx context.Context, in *pb.PullVariableRequest) (*pb.PullVariableResponse, error) {
	var res pb.PullVariableResponse
	if !s.Param.InitStatus {
		res.ModelInitStatus = false
		return &res, nil
	}
	res.ModelInitStatus = true
	res.Model = &pb.Model{}
	res.Model.Version = s.Param.Version
	if s.Param.Version > in.CurrentModelVersion {
		for _, v := range s.Param.NonEmbeddingParam {
			fmt.Println(v.Value)
			res.Model.Param = append(res.Model.Param, common.SerializeTensor(v))
		}
	}
	s.Param.InitStatus = true
	return &res, nil
}

// PullEmbeddingVector pulls embedding vector from server
func (s *Server) PullEmbeddingVector(ctx context.Context, in *pb.PullEmbeddingVectorRequest) (*pb.Tensor, error) {
	if in.Ids == nil {
		return &pb.Tensor{}, nil
	}
	table := s.Param.GetEmbeddingParam(in.Name)
	if table == nil {
		return &pb.Tensor{}, fmt.Errorf("Request embedding Table %s not found in Param", in.Name)
	}
	t := table.GetEmbeddingVectors(in.Ids)
	return common.SerializeTensor(t), nil
}

// PushModel pushes model to server
func (s *Server) PushModel(ctx context.Context, in *pb.Model) (*empty.Empty, error) {
	if s.Param.InitStatus {
		return &empty.Empty{}, nil
	}
	s.lock.Lock()
	err := s.Param.InitFromModelPB(in)
	if err == nil {
		s.Param.InitStatus = true
	}
	s.lock.Unlock()
	return &empty.Empty{}, err
}

// PushEmbeddingInfo pushes embedding info to server
func (s *Server) PushEmbeddingInfo(ctx context.Context, in *pb.Model) (*empty.Empty, error) {
	s.lock.Lock()
	err := s.Param.InitFromModelPB(in)
	s.lock.Unlock()
	return &empty.Empty{}, err
}

// PushGradient pushes gradient to server
func (s *Server) PushGradient(ctx context.Context, in *pb.PushGradientRequest) (*pb.PushGradientResponse, error) {
	// TODO(qijun) only support async now
	var res pb.PushGradientResponse
	var grads []*common.Tensor
	for _, gradPB := range in.Gradients {
		grad := common.DeserializeTensorPB(gradPB)
		grads = append(grads, grad)
	}
	err := s.Opt.ApplyGradients(grads, s.Param)
	s.Param.Version += int32(1)
	res.Accepted = true
	res.ModelVersion = s.Param.Version
	return &res, err
}

// CreateServer creates a PS server and starts the serving. Set serverDone when finishes.
func CreateServer(address string, PSID int32, opt string, lr float32, serverDone chan bool) (*Server, *grpc.Server) {
	lis, err := net.Listen("tcp", address)
	if err != nil {
		log.Fatalf("failed to start PS: %v", err)
	}
	// TODO: set maxReceiveMessageSize (default is 4M, too small for elasticdl), maxConcurrentStreams
	grpcServer := grpc.NewServer()
	s := NewServer(PSID, opt, lr)
	pb.RegisterPserverServer(grpcServer, s)
	go startServe(grpcServer, lis, serverDone)
	return s, grpcServer
}

func startServe(server *grpc.Server, lis net.Listener, serverDone chan bool) {
	err := server.Serve(lis)
	if err != nil {
		log.Fatalf("GRPC failed to serve: %v", err)
	}
	serverDone <- true
}
