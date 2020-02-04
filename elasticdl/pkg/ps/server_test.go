package ps

import (
	"context"
	"elasticdl.org/elasticdl/pkg/common"
	pb "elasticdl.org/elasticdl/pkg/proto"
	"fmt"
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

func createClient() (pb.PserverClient, context.Context, *grpc.ClientConn, context.CancelFunc) {
	conn, err := grpc.Dial(ADDR, grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	c := pb.NewPserverClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	return c, ctx, conn, cancel
}

func TestPushModel(t *testing.T) {
	// Create a PS server
	serverDone := make(chan bool)
	s := CreateServer(ADDR, 0, "SGD", 0.1, serverDone)
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
	// Create a PS server
	serverDone := make(chan bool)
	s := CreateServer(ADDR, 0, "SGD", 0.1, serverDone)
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
		t.Errorf("Failed to push embedding vector info")
	}

	assert.Contains(t, s.Param.EmbeddingParam, "e1")
	assert.Equal(t, int64(2), s.Param.GetEmbeddingParam("e1").Dim)
}

func TestPullVariable(t *testing.T) {
	// Create a PS server
	serverDone := make(chan bool)
	s := CreateServer(ADDR, 0, "SGD", 0.1, serverDone)
	client, ctx, conn, cancel := createClient()
	defer conn.Close()
	defer cancel()

	var request1 pb.Model
	// non embedding param
	a := make([]float32, 6)
	b := make([]float32, 6)
	for i := 0; i < 6; i++ {
		a[i] = rand.Float32()
		b[i] = rand.Float32()
	}
	d := []int64{2, 3}
	t1 := common.Tensor{"t1", a, d, nil}
	t2 := common.Tensor{"t2", b, d, nil}

	fmt.Println(a)
	fmt.Println(b)

	request1.Param = append(request1.Param, common.SerializeTensor(&t1))
	request1.Param = append(request1.Param, common.SerializeTensor(&t2))

	_, err1 := client.PushModel(ctx, &request1)

	if err1 != nil {
		t.Errorf("Failed to push model")
	}

	var request2 pb.PullVariableRequest
	request2.CurrentModelVersion = -1

	res, err2 := client.PullVariable(ctx, &request2)
	if err2 != nil {
		t.Errorf("Failed to pull variable")
	}

	assert.True(t, res.ModelInitStatus)

	p := NewParameter()
	p.InitFromModelPB(res.Model)

	assert.Equal(t, int32(0), p.Version)
	assert.Equal(t, 2, len(p.NonEmbeddingParam))
	assert.Contains(t, p.NonEmbeddingParam, "t1")
	assert.Contains(t, p.NonEmbeddingParam, "t2")
	assert.True(t, common.CompareFloatArray(p.GetNonEmbeddingParam("t1").Value, a, 0.0001))
	assert.True(t, common.CompareFloatArray(p.GetNonEmbeddingParam("t2").Value, b, 0.0001))
}

func TestPullEmbeddingVector(t *testing.T) {
	// Create a PS server
	serverDone := make(chan bool)
	s := CreateServer(ADDR, 0, "SGD", 0.1, serverDone)
	client, ctx, conn, cancel := createClient()
	defer conn.Close()
	defer cancel()

	var request1 pb.Model
	// embedding table info
	var epb pb.EmbeddingTableInfo
	epb.Name = "e1"
	epb.Dim = 2
	epb.Initializer = "zero"
	request1.EmbeddingTableInfo = append(request1.EmbeddingTableInfo, &epb)

	_, err1 := client.PushEmbeddingInfo(ctx, &request1)
	if err1 != nil {
		t.Errorf("Failed to push embedding vector info")
	}

	var request2 pb.PullEmbeddingVectorRequest
	ids := []int64{1, 3, 5}
	request2.Name = "e1"
	request2.Ids = ids

	res, err2 := client.PullEmbeddingVector(ctx, &request2)
	if err2 != nil {
		t.Errorf("Failed to pull embedding vector")
	}

	assert.Contains(t, s.Param.EmbeddingParam, "e1")
	tensor := common.DeserializeTensorPB(res)
	assert.Equal(t, "e1", tensor.Name)
	assert.Equal(t, ids, tensor.Indices)
	assert.Equal(t, 6, len(tensor.Value))
}

func TestPushGradient(t *testing.T) {
	// Create a PS server
	serverDone := make(chan bool)
	s := CreateServer(ADDR, 0, "SGD", 0.1, serverDone)
	client, ctx, conn, cancel := createClient()
	defer conn.Close()
	defer cancel()
	request := pb.PushGradientRequest{}
	_, err := client.PushGradient(ctx, &request)
	if err != nil {
		t.Errorf("Failed to pull embedding vector")
	}
}
