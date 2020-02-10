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

// MasterClient contains attributes to call master GRPC services
type MasterClient struct {
	client     pb.MasterClient
	context    context.Context
	clientConn *grpc.ClientConn
}

func (c *MasterClient) reportVersion(modelVersion int32) {
	var request pb.ReportVersionRequest
	request.ModelVersion = modelVersion
	c.client.ReportVersion(c.context, &request)
}

func (c *MasterClient) closeConn() {
	c.clientConn.Close()
}

// Server defines servicer of ps
type Server struct {
	pb.PserverServer
	Param           *Parameter
	Opt             Optimizer
	masterClient    *MasterClient
	evaluationSteps int32
	ID              int // a zero-based successive integer number
	lock            sync.Mutex
	versionLock     sync.Mutex
}

func createMasterClient(masterAddr string) *MasterClient {
	if masterAddr == "" {
		return nil
	}
	conn, err := grpc.Dial(masterAddr, grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		log.Fatalf("failed to connect to master: %v", err)
	}
	client := pb.NewMasterClient(conn)
	return &MasterClient{
		client:     client,
		context:    context.Background(),
		clientConn: conn,
	}
}

// NewServer creates a Server instance
func NewServer(ID int, opt string, lr float32, masterAddr string, evaluationSteps int32) *Server {
	client := createMasterClient(masterAddr)
	return &Server{
		Param:           NewParameter(),
		Opt:             NewOptimizer(opt, lr),
		ID:              ID,
		masterClient:    client,
		evaluationSteps: evaluationSteps}
}

func (s *Server) reportModelVersionIfNeeded(modelVersion int32) {
	if s.evaluationSteps > 0 && modelVersion%s.evaluationSteps == 0 && s.masterClient != nil {
		s.masterClient.reportVersion(modelVersion)
	}
}

// PullVariable pulls variable from server
func (s *Server) PullVariable(ctx context.Context, in *pb.PullVariableRequest) (*pb.PullVariableResponse, error) {
	// TODO(qijun) only support async now
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
	s.lock.Lock()
	var err error
	if !s.Param.InitStatus {
		err = s.Param.InitFromModelPB(in)
		if err == nil {
			s.Param.InitStatus = true
		}
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
	s.versionLock.Lock()
	s.Param.Version += int32(1)
	s.versionLock.Unlock()
	s.reportModelVersionIfNeeded(s.Param.Version)
	res.Accepted = true
	res.ModelVersion = s.Param.Version
	return &res, err
}

// Run creates a grpc server and starts the serving. Set serverDone when finishes.
func (s *Server) Run(address string, serverDone chan bool) *grpc.Server {
	lis, err := net.Listen("tcp", address)
	if err != nil {
		log.Fatalf("failed to start PS: %v", err)
	}
	// TODO: set maxReceiveMessageSize (default is 4M, too small for elasticdl), maxConcurrentStreams
	grpcServer := grpc.NewServer()
	pb.RegisterPserverServer(grpcServer, s)
	go startServe(grpcServer, lis, serverDone, s.masterClient)
	return grpcServer
}

func startServe(server *grpc.Server, lis net.Listener, serverDone chan bool, masterClient *MasterClient) {
	defer masterClient.closeConn()
	err := server.Serve(lis)
	if err != nil {
		log.Fatalf("GRPC failed to serve: %v", err)
	}
	serverDone <- true
}
