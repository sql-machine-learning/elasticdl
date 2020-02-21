package ps

import (
	"context"
	"elasticdl.org/elasticdl/common"
	"elasticdl.org/elasticdl/proto"
	"github.com/golang/protobuf/ptypes/empty"
	"github.com/stretchr/testify/assert"
	"github.com/tensorflow/tensorflow/tensorflow/go/core/framework/tensor_go_proto"
	"google.golang.org/grpc"
	"log"
	"math/rand"
	"net"
	"testing"
	"time"
)

const (
	ADDR string = "localhost:12345"
)

type masterServer struct {
	proto.UnimplementedMasterServer
	address      string
	modelVersion int32
	server       *grpc.Server
}

func (s *masterServer) run() {
	lis, err := net.Listen("tcp", s.address)
	if err != nil {
		log.Fatalf("failed to start Master: %v", err)
	}
	s.server = grpc.NewServer()
	proto.RegisterMasterServer(s.server, s)
	go s.startServe(lis)
}

func (s *masterServer) startServe(lis net.Listener) {
	s.server.Serve(lis)
}

func (s *masterServer) stop() {
	s.server.Stop()
}

// ReportVersion grpc service
func (s *masterServer) ReportVersion(ctx context.Context, in *proto.ReportVersionRequest) (*empty.Empty, error) {
	var res empty.Empty
	if in.ModelVersion > s.modelVersion {
		s.modelVersion = in.ModelVersion
	}
	return &res, nil
}

func newMasterServer(addr string) *masterServer {
	server := masterServer{modelVersion: int32(0), address: addr}
	return &server
}

func createClient() (proto.PserverClient, context.Context, *grpc.ClientConn, context.CancelFunc) {
	conn, err := grpc.Dial(ADDR, grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	c := proto.NewPserverClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	return c, ctx, conn, cancel
}

func TestMasterClient(t *testing.T) {
	// Create a Master server
	masterAddr := "localhost:12368"
	masterServer := newMasterServer(masterAddr)
	masterServer.run()
	// New a PS server
	s := NewServer(0, "SGD", "learning_rate=0.1;momentum=0.0;nesterov=false;", masterAddr, 0)

	version := int32(2)
	s.masterClient.reportVersion(version)
	assert.Equal(t, masterServer.modelVersion, version)
	s.masterClient.reportVersion(int32(1))
	assert.Equal(t, masterServer.modelVersion, version)
	version = int32(22)
	s.masterClient.reportVersion(version)
	assert.Equal(t, masterServer.modelVersion, version)

	masterServer.stop()
	s.masterClient.closeConn()
}

func TestPushModel(t *testing.T) {
	// Create a PS server
	serverDone := make(chan bool)
	s := NewServer(0, "SGD", "learning_rate=0.1;momentum=0.0;nesterov=false;", "", 0)
	gs := s.Run(ADDR, serverDone)
	client, ctx, conn, cancel := createClient()
	defer conn.Close()
	defer cancel()

	var request = &proto.Model{
		DenseParameters: make(map[string]*tensor_go_proto.TensorProto),
		EmbeddingTables: make(map[string]*proto.IndexedSlicesProto),
		EmbeddingTableInfos: []*proto.EmbeddingTableInfo{&proto.EmbeddingTableInfo{
			Name:        "e1",
			Dim:         2,
			Initializer: "zero",
		}},
	}
	// dense embedding param
	a := make([]float32, 10)
	b := make([]float32, 10)
	for i := 0; i < 10; i++ {
		a[i] = rand.Float32()
		b[i] = rand.Float32()
	}
	d := []int64{2, 5}
	t1 := common.NewTensor(a, d) // t1
	t2 := common.NewTensor(b, d) // t2

	request.DenseParameters["t1"] = t1.SerializeToTensorProto()
	request.DenseParameters["t2"] = t2.SerializeToTensorProto()

	_, err := client.PushModel(ctx, request)

	if err != nil {
		t.Errorf("Failed to push model")
	}

	assert.True(t, s.Model.Initialized)
	assert.Len(t, s.Model.DenseParameters, 2)
	assert.Contains(t, s.Model.DenseParameters, "t1")
	assert.Contains(t, s.Model.DenseParameters, "t2")
	assert.True(t, common.CompareFloatArray(a, common.Slice(s.Model.GetDenseParameter("t1")).([]float32), 0.0001))
	assert.True(t, common.CompareFloatArray(b, common.Slice(s.Model.GetDenseParameter("t2")).([]float32), 0.0001))
	assert.Contains(t, s.Model.EmbeddingTables, "e1")
	assert.Equal(t, int64(2), s.Model.GetEmbeddingTable("e1").Dim)
	gs.Stop()
}

func TestPullEmbeddingVectors(t *testing.T) {
	// Create a PS server
	serverDone := make(chan bool)
	s := NewServer(0, "SGD", "learning_rate=0.1;momentum=0.0;nesterov=false;", "", 0)
	gs := s.Run(ADDR, serverDone)
	client, ctx, conn, cancel := createClient()
	defer conn.Close()
	defer cancel()

	var request = &proto.Model{
		DenseParameters: make(map[string]*tensor_go_proto.TensorProto),
		EmbeddingTables: make(map[string]*proto.IndexedSlicesProto),
		EmbeddingTableInfos: []*proto.EmbeddingTableInfo{&proto.EmbeddingTableInfo{
			Name:        "e1",
			Dim:         10,
			Initializer: "zero",
			Dtype:       common.Float32,
		}},
	}

	// dense embedding param
	a := make([]float32, 10)
	b := make([]float32, 10)
	c := make([]float32, 10)
	for i := 0; i < 10; i++ {
		a[i] = rand.Float32()
		b[i] = rand.Float32()
		c[i] = rand.Float32()
	}
	d := []int64{2, 5}
	t1 := common.NewTensor(a, d) // t1
	t2 := common.NewTensor(b, d) // t2

	ed := []int64{1, 10}
	e1 := common.NewIndexedSlices(common.NewTensor(c, ed), []int64{1})

	request.DenseParameters["t1"] = t1.SerializeToTensorProto()
	request.DenseParameters["t2"] = t2.SerializeToTensorProto()
	request.EmbeddingTables["e1"] = e1.SerializeToIndexedSlicesProto()

	client.PushModel(ctx, request)

	pr := &proto.PullEmbeddingVectorsRequest{
		Name: "e1",
		Ids:  []int64{1},
	}

	resp, _ := client.PullEmbeddingVectors(ctx, pr)
	assert.True(t, common.CompareFloatArray(c, common.Slice(common.DeserializeFromTensorProto(resp)).([]float32), 0.0001))
	gs.Stop()
}

func TestPullDenseParameters(t *testing.T) {
	// Create a PS server
	serverDone := make(chan bool)
	s := NewServer(0, "SGD", "learning_rate=0.1;momentum=0.0;nesterov=false;", "", 0)
	gs := s.Run(ADDR, serverDone)
	client, ctx, conn, cancel := createClient()
	defer conn.Close()
	defer cancel()

	var request = &proto.Model{
		DenseParameters: make(map[string]*tensor_go_proto.TensorProto),
		EmbeddingTables: make(map[string]*proto.IndexedSlicesProto),
		EmbeddingTableInfos: []*proto.EmbeddingTableInfo{&proto.EmbeddingTableInfo{
			Name:        "e1",
			Dim:         10,
			Initializer: "zero",
			Dtype:       common.Float32,
		}},
	}

	// dense embedding param
	a := make([]float32, 10)
	b := make([]float32, 10)
	c := make([]float32, 10)
	for i := 0; i < 10; i++ {
		a[i] = rand.Float32()
		b[i] = rand.Float32()
		c[i] = rand.Float32()
	}
	d := []int64{2, 5}
	t1 := common.NewTensor(a, d) // t1
	t2 := common.NewTensor(b, d) // t2

	ed := []int64{1, 10}
	e1 := common.NewIndexedSlices(common.NewTensor(c, ed), []int64{1})

	request.DenseParameters["t1"] = t1.SerializeToTensorProto()
	request.DenseParameters["t2"] = t2.SerializeToTensorProto()
	request.EmbeddingTables["e1"] = e1.SerializeToIndexedSlicesProto()

	client.PushModel(ctx, request)

	pr := &proto.PullDenseParametersRequest{
		Version: 0,
	}

	resp, _ := client.PullDenseParameters(ctx, pr)
	assert.Equal(t, true, resp.Initialized)
	assert.Equal(t, int32(0), resp.Version)
	assert.True(t, common.CompareFloatArray(a, common.Slice(common.DeserializeFromTensorProto(resp.DenseParameters["t1"])).([]float32), 0.0001))
	assert.True(t, common.CompareFloatArray(b, common.Slice(common.DeserializeFromTensorProto(resp.DenseParameters["t2"])).([]float32), 0.0001))
	gs.Stop()
}

func TestPushGradients(t *testing.T) {
	// Create a PS server
	serverDone := make(chan bool)
	s := NewServer(0, "SGD", "learning_rate=0.1;momentum=0.0;nesterov=false;", "", 0)
	gs := s.Run(ADDR, serverDone)
	client, ctx, conn, cancel := createClient()
	defer conn.Close()
	defer cancel()

	var request = &proto.Model{
		DenseParameters: make(map[string]*tensor_go_proto.TensorProto),
		EmbeddingTables: make(map[string]*proto.IndexedSlicesProto),
		EmbeddingTableInfos: []*proto.EmbeddingTableInfo{&proto.EmbeddingTableInfo{
			Name:        "e1",
			Dim:         10,
			Initializer: "zero",
			Dtype:       common.Float32,
		}},
	}

	// dense embedding param
	a := make([]float32, 10)
	b := make([]float32, 10)
	c := make([]float32, 10)
	for i := 0; i < 10; i++ {
		a[i] = rand.Float32()
		b[i] = rand.Float32()
		c[i] = rand.Float32()
	}
	d := []int64{2, 5}
	t1 := common.NewTensor(a, d) // t1
	t2 := common.NewTensor(b, d) // t2

	ed := []int64{1, 10}
	e1 := common.NewIndexedSlices(common.NewTensor(c, ed), []int64{1})

	request.DenseParameters["t1"] = t1.SerializeToTensorProto()
	request.DenseParameters["t2"] = t2.SerializeToTensorProto()
	request.EmbeddingTables["e1"] = e1.SerializeToIndexedSlicesProto()

	client.PushModel(ctx, request)

	_, err := client.PushGradients(ctx, request)
	if err != nil {
		t.Errorf("Failed to push embedding vector info")
	}

	expectedt1 := make([]float32, 10, 10)
	expectedt2 := make([]float32, 10, 10)
	expectede1 := make([]float32, 10, 10)

	for i := 0; i < 10; i++ {
		expectedt1[i] = a[i] - 0.1*a[i]
		expectedt2[i] = b[i] - 0.1*b[i]
		expectede1[i] = c[i] - 0.1*c[i]
	}

	assert.True(t, common.CompareFloatArray(expectedt1, common.Slice(s.Model.GetDenseParameter("t1")).([]float32), 0.0001))
	assert.True(t, common.CompareFloatArray(expectedt2, common.Slice(s.Model.GetDenseParameter("t2")).([]float32), 0.0001))
	assert.True(t, common.CompareFloatArray(expectede1, common.Slice(s.Model.GetEmbeddingTable("e1").GetEmbeddingVector(1)).([]float32), 0.0001))
	gs.Stop()
}
