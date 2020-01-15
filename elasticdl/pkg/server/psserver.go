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
	pb.PserverServer
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

// CreateServer creates a PS server and starts the serving. Set serverDone when finishes.
func CreateServer(address string, serverDone chan bool) {
	lis, err := net.Listen("tcp", address)
	if err != nil {
		log.Fatalf("failed to start PS: %v", err)
	}
	// TODO: set maxReceiveMessageSize (default is 4M, too small for elasticdl), maxConcurrentStreams
	grpcServer := grpc.NewServer()
	pb.RegisterPserverServer(grpcServer, &psServer{})
	go startServe(grpcServer, lis, serverDone)
}

func startServe(server *grpc.Server, lis net.Listener, serverDone chan bool) {
	err := server.Serve(lis)
	if err != nil {
		log.Fatalf("GRPC failed to serve: %v", err)
	}
	serverDone <- true
}

func main() {
	flag.Parse()
	address := fmt.Sprintf("localhost:%d", *port)
	serverDone := make(chan bool)
	CreateServer(address, serverDone)
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
