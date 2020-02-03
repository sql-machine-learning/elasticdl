package ps

import (
	"context"
	"elasticdl.org/elasticdl/pkg/common"
	pb "elasticdl.org/elasticdl/pkg/proto"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	"log"
	"math/rand"
	"os"
	"testing"
	"time"
)

const (
	ADDR string = "localhost:12345"
)

var s *Server

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
	s = CreateServer(ADDR, 0, "SGD", 0.1, serverDone)

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

	var request pb.Model
	// non embedding param
	a := make([]float32, 10)
	b := make([]float32, 10)
	for i := 0; i < 10; i++ {
		a[i] = rand.Float32()
		b[i] = rand.Float32()
	}
	d := []int64{2, 5}
	t1 := common.Tensor{"t1", a, d, nil}
	t2 := common.Tensor{"t2", b, d, nil}

	request.Param = append(request.Param, common.SerializeTensor(&t1))
	request.Param = append(request.Param, common.SerializeTensor(&t2))

	_, err := client.PushModel(ctx, &request)

	if err != nil {
		t.Errorf("Failed to push model")
	}

	assert.True(t, s.Param.InitStatus)
	assert.Len(t, s.Param.NonEmbeddingParam, 2)
	assert.Contains(t, s.Param.NonEmbeddingParam, "t1")
	assert.Contains(t, s.Param.NonEmbeddingParam, "t2")
	assert.True(t, common.CompareFloatArray(a, s.Param.GetNonEmbeddingParam("t1").Value, 0.001))
	assert.True(t, common.CompareFloatArray(b, s.Param.GetNonEmbeddingParam("t2").Value, 0.001))
}

func TestPushEmbeddingInfo(t *testing.T) {
	client, ctx, conn, cancel := createClient()
	defer conn.Close()
	defer cancel()

	var request pb.Model
	// embedding table info
	var epb pb.EmbeddingTableInfo
	epb.Name = "e1"
	epb.Dim = 2
	epb.Initializer = "zero"
	request.EmbeddingTableInfo = append(request.EmbeddingTableInfo, &epb)

	_, err := client.PushEmbeddingInfo(ctx, &request)
	if err != nil {
		t.Errorf("Failed to pull embedding vector")
	}

	assert.Contains(t, s.Param.EmbeddingParam, "e1")
	assert.Equal(t, int64(2), s.Param.GetEmbeddingParam("e1").Dim)
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
