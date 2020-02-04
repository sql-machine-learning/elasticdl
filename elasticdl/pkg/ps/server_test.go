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
	s, gs := CreateServer(ADDR, 0, "SGD", 0.1, serverDone)
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
	gs.Stop()
}

func TestPushEmbeddingInfo(t *testing.T) {
	// Create a PS server
	serverDone := make(chan bool)
	s, gs := CreateServer(ADDR, 0, "SGD", 0.1, serverDone)
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
	gs.Stop()
}

func TestPullVariable(t *testing.T) {
	// Create a PS server
	serverDone := make(chan bool)
	s, gs := CreateServer(ADDR, 0, "SGD", 0.1, serverDone)
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

	assert.Contains(t, s.Param.NonEmbeddingParam, "t1")
	assert.Contains(t, s.Param.NonEmbeddingParam, "t2")

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
	gs.Stop()
}

func TestPullEmbeddingVector(t *testing.T) {
	// Create a PS server
	serverDone := make(chan bool)
	s, gs := CreateServer(ADDR, 0, "SGD", 0.1, serverDone)
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
	gs.Stop()
}

func TestPushGradient(t *testing.T) {
	// Create a PS server
	serverDone := make(chan bool)
	s, gs := CreateServer(ADDR, 0, "SGD", 0.1, serverDone)
	client, ctx, conn, cancel := createClient()
	defer conn.Close()
	defer cancel()

	// non embedding param
	v1 := []float32{10.0, 20.0, 30.0}
	v2 := []float32{20.0, 40.0, 60.0}
	d := []int64{1, 3}
	t1 := common.Tensor{"t1", v1, d, nil}
	t2 := common.Tensor{"t2", v2, d, nil}

	// embedding param info
	var epb pb.EmbeddingTableInfo
	epb.Name = "e1"
	epb.Dim = 2
	epb.Initializer = "zero"

	ev := []float32{1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0}
	ed := []int64{4, 2}
	ei := []int64{1, 2, 3, 4}
	e1 := common.Tensor{"e1", ev, ed, ei}

	// push model request
	var request1 pb.Model
	request1.EmbeddingTableInfo = append(request1.EmbeddingTableInfo, &epb)
	request1.Param = append(request1.Param, common.SerializeTensor(&t1))
	request1.Param = append(request1.Param, common.SerializeTensor(&t2))
	request1.Param = append(request1.Param, common.SerializeTensor(&e1))

	_, err1 := client.PushModel(ctx, &request1)
	if err1 != nil {
		t.Errorf("Failed to push model")
	}

	gv1 := []float32{1.0, 2.0, 3.0}
	gv2 := []float32{2.0, 4.0, 6.0}
	g1 := common.Tensor{"t1", gv1, d, nil}
	g2 := common.Tensor{"t2", gv2, d, nil}

	egv1 := []float32{1.0, 1.0, 1.0, 2.0, 2.0, 2.0}
	egd1 := []int64{3, 2}
	egi1 := []int64{3, 1, 3}
	eg1 := common.Tensor{"e1", egv1, egd1, egi1}

	var request2 pb.PushGradientRequest
	request2.ModelVersion = 0
	request2.Gradients = append(request2.Gradients, common.SerializeTensor(&g1))
	request2.Gradients = append(request2.Gradients, common.SerializeTensor(&g2))
	request2.Gradients = append(request2.Gradients, common.SerializeTensor(&eg1))

	res1, err2 := client.PushGradient(ctx, &request2)
	if err2 != nil {
		t.Errorf("Failed to pull gradients")
	}

	assert.True(t, res1.Accepted)
	assert.Equal(t, int32(1), res1.ModelVersion)

	assert.Contains(t, s.Param.NonEmbeddingParam, "t1")
	assert.Contains(t, s.Param.NonEmbeddingParam, "t2")
	assert.Contains(t, s.Param.EmbeddingParam, "e1")
	exptV1 := []float32{9.9, 19.8, 29.7}
	exptV2 := []float32{19.8, 39.6, 59.4}
	assert.True(t, common.CompareFloatArray(s.Param.GetNonEmbeddingParam("t1").Value, exptV1, 0.0001))
	assert.True(t, common.CompareFloatArray(s.Param.GetNonEmbeddingParam("t2").Value, exptV2, 0.0001))

	expGV1 := []float32{1.0, 2.0, 2.9, 3.8, 5.0, 6.0, 6.7, 7.7}
	actGV1 := s.Param.GetEmbeddingParam("e1").GetEmbeddingVectors(ei)
	assert.True(t, common.CompareFloatArray(actGV1.Value, expGV1, 0.0001))
	gs.Stop()
}
