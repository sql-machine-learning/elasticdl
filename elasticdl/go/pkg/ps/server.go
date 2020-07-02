// Copyright 2020 The ElasticDL Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ps

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"path"
	"sync"

	"elasticdl.org/elasticdl/pkg/proto"
	"github.com/golang/protobuf/ptypes/empty"
	"github.com/tensorflow/tensorflow/tensorflow/go/core/framework/tensor_go_proto"
	"google.golang.org/grpc"
)

const (
	maxSendMessageLength    = 256 * 1024 * 1024
	maxReceiveMessageLength = 256 * 1024 * 1024
)

// MasterClient contains attributes to call master GRPC services
type MasterClient struct {
	client     proto.MasterClient
	context    context.Context
	clientConn *grpc.ClientConn
}

func (c *MasterClient) reportVersion(modelVersion int32) {
	var request proto.ReportVersionRequest
	request.ModelVersion = modelVersion
	c.client.ReportVersion(c.context, &request)
}

func (c *MasterClient) closeConn() {
	c.clientConn.Close()
}

// Server defines servicer of ps
type Server struct {
	proto.UnimplementedPserverServer
	Model                 *Model
	Opt                   Optimizer
	masterClient          *MasterClient
	evaluationStep        int
	checkpointDirForInit  string
	checkpointDir         string
	checkpointStep        int
	keepCheckpointMax     int
	numPsPods             int
	lrStalenessModulation bool
	ID                    int // a zero-based successive integer number
	lock                  sync.Mutex
	versionLock           sync.Mutex
	savedCheckpointDirs   []string
}

func createMasterClient(masterAddr string) *MasterClient {
	if masterAddr == "" {
		return nil
	}
	conn, err := grpc.Dial(masterAddr, grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		log.Fatalf("failed to connect to master: %v", err)
	}
	client := proto.NewMasterClient(conn)
	return &MasterClient{
		client:     client,
		context:    context.Background(),
		clientConn: conn,
	}
}

// NewServer creates a Server instance
func NewServer(ID int, optType string, optArgs string, masterAddr string,
	evaluationStep int, checkpointDirForInit string,
	checkpointDir string, checkpointStep int, keepCheckpointMax int, numPsPods int,
	lrStalenessModulation bool) *Server {
	var ps Server
	if checkpointDirForInit != "" {
		var err error
		ps.Model, err = LoadModelFromCheckpoint(checkpointDirForInit, ID, numPsPods)
		ps.Model.Initialized = true
		if err != nil {
			log.Fatalf("failed to load from checkpoint: %v", err)
		}
	} else {
		ps.Model = NewModel()
	}

	var err error
	ps.Opt, err = NewOptimizer(optType, optArgs)
	if err != nil {
		log.Fatalf("failed to create PS server: %v", err)
	}
	ps.ID = ID
	ps.masterClient = createMasterClient(masterAddr)
	ps.evaluationStep = evaluationStep
	ps.checkpointDirForInit = checkpointDirForInit
	ps.checkpointDir = checkpointDir
	ps.checkpointStep = checkpointStep
	ps.keepCheckpointMax = keepCheckpointMax
	ps.numPsPods = numPsPods
	ps.lrStalenessModulation = lrStalenessModulation
	return &ps
}

func (s *Server) reportModelVersionIfNeeded(modelVersion int) {
	if s.evaluationStep > 0 && modelVersion%s.evaluationStep == 0 && s.masterClient != nil {
		s.masterClient.reportVersion(int32(modelVersion))
	}
}

func (s *Server) saveCheckpointIfNeeded(modelVersion int) {
	if s.checkpointDir != "" && s.checkpointStep != 0 && modelVersion%s.checkpointStep == 0 {
		checkpointVersionDir := path.Join(s.checkpointDir, fmt.Sprintf("version-%d", modelVersion))
		s.savedCheckpointDirs = append(s.savedCheckpointDirs, checkpointVersionDir)
		SaveModelToCheckpoint(checkpointVersionDir, s.Model, s.ID, s.numPsPods)
		if s.ID == 0 {
			if len(s.savedCheckpointDirs) > s.keepCheckpointMax {
				deletedDir := s.savedCheckpointDirs[0]
				s.savedCheckpointDirs = s.savedCheckpointDirs[1:]
				os.RemoveAll(deletedDir)
			}
		}
	}
}

// PullDenseParameters pulls dense parameter from server
func (s *Server) PullDenseParameters(ctx context.Context, in *proto.PullDenseParametersRequest) (*proto.PullDenseParametersResponse, error) {
	if !s.Model.Initialized {
		return &proto.PullDenseParametersResponse{Initialized: false}, nil
	}
	denseParamPB := make(map[string]*tensor_go_proto.TensorProto)
	if s.Model.Version >= in.Version {
		for name, tensor := range s.Model.DenseParameters {
			denseParamPB[name] = tensor.SerializeToTensorProto()
		}
	}
	var resp = proto.PullDenseParametersResponse{
		Initialized:     true,
		Version:         s.Model.Version,
		DenseParameters: denseParamPB,
	}
	return &resp, nil
}

// PullEmbeddingVectors pulls sparse parameter from server
func (s *Server) PullEmbeddingVectors(ctx context.Context, in *proto.PullEmbeddingVectorsRequest) (*tensor_go_proto.TensorProto, error) {
	if in.Ids == nil {
		return &tensor_go_proto.TensorProto{}, nil
	}
	table := s.Model.GetEmbeddingTable(in.Name)
	if table == nil {
		return &tensor_go_proto.TensorProto{}, fmt.Errorf("Request embedding Table %s not found in Param", in.Name)
	}
	t := table.GetEmbeddingVectors(in.Ids)
	return t.SerializeToTensorProto(), nil
}

// PushGradients push gradients to server
func (s *Server) PushGradients(ctx context.Context, in *proto.PushGradientsRequest) (*proto.PushGradientsResponse, error) {
	// TODO: only support async now
	var lr = float32(1.0)
	if s.lrStalenessModulation && s.Model.Version > in.Gradients.Version {
		staleness := s.Model.Version - in.Gradients.Version
		lr = lr / float32(staleness)
	}
	if in.LearningRate > 0.0 {
		lr = lr * in.LearningRate
	} else {
		lr = lr * s.Opt.GetLR()
	}
	err := s.Opt.ApplyGradients(in.Gradients, s.Model, lr)
	if err != nil {
		var resp = proto.PushGradientsResponse{
			Accepted: false,
			Version:  s.Model.Version,
		}
		return &resp, err
	}
	s.versionLock.Lock()
	s.Model.Version += int32(1)
	s.saveCheckpointIfNeeded(int(s.Model.Version))
	s.versionLock.Unlock()
	s.reportModelVersionIfNeeded(int(s.Model.Version))
	var resp = proto.PushGradientsResponse{
		Accepted: true,
		Version:  s.Model.Version,
	}
	return &resp, nil
}

// PushModel push Model to server
func (s *Server) PushModel(ctx context.Context, in *proto.Model) (*empty.Empty, error) {
	s.lock.Lock()
	var err error
	if !s.Model.Initialized {
		err = s.Model.InitFromModelPB(in)
		s.Opt.InitOptimizer(in)
		if err == nil {
			s.Model.Initialized = true
		}
	}
	s.lock.Unlock()
	return &empty.Empty{}, err
}

// PushEmbeddingTableInfos pushes embedding table infos to server
func (s *Server) PushEmbeddingTableInfos(ctx context.Context, in *proto.Model) (*empty.Empty, error) {
	s.lock.Lock()
	err := s.Model.InitFromModelPB(in)
	s.Opt.InitOptimizer(in)
	s.lock.Unlock()
	return &empty.Empty{}, err
}

// Run creates a grpc server and starts the serving. Set serverDone when finishes.
func (s *Server) Run(address string, concurrentStreams int, serverDone chan bool) *grpc.Server {
	lis, err := net.Listen("tcp", address)
	if err != nil {
		log.Fatalf("failed to start PS: %v", err)
	}
	grpcServer := grpc.NewServer(grpc.MaxRecvMsgSize(maxReceiveMessageLength),
		grpc.MaxSendMsgSize(maxSendMessageLength),
		grpc.MaxConcurrentStreams(uint32(concurrentStreams)))
	proto.RegisterPserverServer(grpcServer, s)
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
