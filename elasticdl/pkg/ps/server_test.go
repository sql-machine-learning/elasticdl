package main

import "testing"
import "os"
import "context"
import "log"
import "time"
import "google.golang.org/grpc"
import pb "elasticdl.org/elasticdl/pkg/proto"

const (
	ADDR string = "localhost:12345"
)

func createClient() (pb.PserverClient, context.Context, *grpc.ClientConn, context.CancelFunc) {
	conn, err := grpc.Dial(ADDR, grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	c := pb.NewPserverClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	return c, ctx, conn, cancel
}

func TestMain(m *testing.M) {
	// Create a PS server
	serverDone := make(chan bool)
	CreateServer(ADDR, serverDone)

	result := m.Run()

	os.Exit(result)
}

func TestPullVariable(t *testing.T) {
	client, ctx, conn, cancel := createClient()
	defer conn.Close()
	defer cancel()
	request := pb.PullVariableRequest{}
	_, err := client.PullVariable(ctx, &request)
	if err != nil {
		t.Errorf("Failed to pull variable")
	}
}

func TestPullEmbeddingVector(t *testing.T) {
	client, ctx, conn, cancel := createClient()
	defer conn.Close()
	defer cancel()
	request := pb.PullEmbeddingVectorRequest{}
	_, err := client.PullEmbeddingVector(ctx, &request)
	if err != nil {
		t.Errorf("Failed to pull embedding vector")
	}
}

func TestPushModel(t *testing.T) {
	client, ctx, conn, cancel := createClient()
	defer conn.Close()
	defer cancel()
	request := pb.Model{}
	_, err := client.PushModel(ctx, &request)
	if err != nil {
		t.Errorf("Failed to push model")
	}
}

func TestPushEmbeddingInfo(t *testing.T) {
	client, ctx, conn, cancel := createClient()
	defer conn.Close()
	defer cancel()
	request := pb.Model{}
	_, err := client.PushEmbeddingInfo(ctx, &request)
	if err != nil {
		t.Errorf("Failed to pull embedding vector")
	}
}

func TestPushGradient(t *testing.T) {
	client, ctx, conn, cancel := createClient()
	defer conn.Close()
	defer cancel()
	request := pb.PushGradientRequest{}
	_, err := client.PushGradient(ctx, &request)
	if err != nil {
		t.Errorf("Failed to pull embedding vector")
	}
}
