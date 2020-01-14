package main

import (
	"context"
	pb "elasticdl/pkg/proto"
	"flag"
	"fmt"
	"log"
	"net"
	"time"

	empty "github.com/golang/protobuf/ptypes/empty"
	"google.golang.org/grpc"
)

type psServer struct {
	pb.MasterServer
}

var (
	// TODO: parse more args
	port = flag.Int("port", 10000, "The server port")
)

func (s *psServer) PullVariable(ctx context.Context, in *pb.PullVariableRequest) (*pb.PullVariableResponse, error) {
	// TODO: implement the service.
	return &pb.PullVariableResponse{}, nil
}

func (s *psServer) PullEmbeddingVector(ctx context.Context, in *pb.PullEmbeddingVectorRequest) (*pb.Tensor, error) {
	// TODO: implement the service.
	return &pb.Tensor{}, nil
}

func (s *psServer) PushModel(ctx context.Context, in *pb.Model) (*empty.Empty, error) {
	// TODO: implement the service.
	return &empty.Empty{}, nil
}

func (s *psServer) PushEmbeddingInfo(ctx context.Context, in *pb.Model) (*empty.Empty, error) {
	// TODO: implement the service.
	return &empty.Empty{}, nil
}

func (s *psServer) PushGradient(ctx context.Context, in *pb.PushGradientRequest) (*pb.PushGradientResponse, error) {
	// TODO: implement the service.
	return &pb.PushGradientResponse{}, nil
}

func StartServe(server *grpc.Server, lis net.Listener, serverFailed chan bool) {
	err := server.Serve(lis)
	if err != nil {
		log.Fatalf("GRPC failed to serve: %v", err)
	}
	serverFailed <- true
}

func main() {
	flag.Parse()
	lis, err := net.Listen("tcp", fmt.Sprintf("localhost:%d", *port))
	if err != nil {
		log.Fatalf("failed to start PS: %v", err)
	}
	grpcServer := grpc.NewServer()
	pb.RegisterMasterServer(grpcServer, &psServer{})
	log.Println("PS service started.")
	serverFailed := make(chan bool)
	go StartServe(grpcServer, lis, serverFailed)
	for {
		select {
		case failed := <-serverFailed:
			_ = failed
			break
		default:
			// TODO: check master pod status and break loop if needed
			time.Sleep(time.Second * 30)
		}
	}
}
