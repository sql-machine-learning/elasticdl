package ps

import (
	"context"
	pb "elasticdl.org/elasticdl/pkg/proto"
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
	// TODO: implement the service.
	return &pb.PullVariableResponse{}, nil
}

// PullEmbeddingVector pulls embedding vector from server
func (s *Server) PullEmbeddingVector(ctx context.Context, in *pb.PullEmbeddingVectorRequest) (*pb.Tensor, error) {
	// TODO: implement the service.
	return &pb.Tensor{}, nil
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
	// TODO: implement the service.
	return &pb.PushGradientResponse{}, nil
}

// CreateServer creates a PS server and starts the serving. Set serverDone when finishes.
func CreateServer(address string, PSID int32, opt string, lr float32, serverDone chan bool) *Server {
	lis, err := net.Listen("tcp", address)
	if err != nil {
		log.Fatalf("failed to start PS: %v", err)
	}
	// TODO: set maxReceiveMessageSize (default is 4M, too small for elasticdl), maxConcurrentStreams
	grpcServer := grpc.NewServer()
	s := NewServer(PSID, opt, lr)
	pb.RegisterPserverServer(grpcServer, s)
	go startServe(grpcServer, lis, serverDone)
	return s
}

func startServe(server *grpc.Server, lis net.Listener, serverDone chan bool) {
	err := server.Serve(lis)
	if err != nil {
		log.Fatalf("GRPC failed to serve: %v", err)
	}
	serverDone <- true
}
