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
	port = flag.Int("port", 2222, "The server port")
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

// StartServe starts server serving, and set serverDone when finishes.
func StartServe(server *grpc.Server, lis net.Listener, serverDone chan bool) {
	err := server.Serve(lis)
	if err != nil {
		log.Fatalf("GRPC failed to serve: %v", err)
	}
	serverDone <- true
}

func main() {
	flag.Parse()
	lis, err := net.Listen("tcp", fmt.Sprintf("localhost:%d", *port))
	if err != nil {
		log.Fatalf("failed to start PS: %v", err)
	}
	// TODO: set maxReceiveMessageSize (default is 4M, too small for elasticdl), maxConcurrentStreams
	grpcServer := grpc.NewServer()
	pb.RegisterMasterServer(grpcServer, &psServer{})
	serverDone := make(chan bool)
	go StartServe(grpcServer, lis, serverDone)
	log.Println("PS service started.")
	for {
		select {
		case done := <-serverDone:
			_ = done
			break
		default:
			// TODO: check master pod status and break loop if needed
			time.Sleep(time.Second * 30)
		}
	}
}
