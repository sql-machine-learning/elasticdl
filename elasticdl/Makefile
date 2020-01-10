PYTHON = python

EDL_PROTO_FILE = elasticdl/proto/elasticdl.proto
TENSOR_DTYPE_PROTO_FILE = elasticdl/proto/tensor_dtype.proto

GO_FILE_DIR = elasticdl/pkg/proto
GO_PB_FILE = $(GO_FILE_DIR)/elasticdl.pb.go
TENSOR_DTYPE_GO_FILE  = $(GO_FILE_DIR)/tensor_dtype.pb.go

all: python_pb $(GO_PB_FILE) $(TENSOR_DTYPE_GO_FILE)

python_pb:
	$(PYTHON) -m grpc_tools.protoc -I . $(TENSOR_DTYPE_PROTO_FILE) --python_out=.
	$(PYTHON) -m grpc_tools.protoc -I . $(EDL_PROTO_FILE) --python_out=. --grpc_python_out=.

$(GO_PB_FILE): $(EDL_PROTO_FILE)
	mkdir -p $(GO_FILE_DIR)
	protoc -I . $< --go_out=plugins=grpc:.

$(TENSOR_DTYPE_GO_FILE): $(TENSOR_DTYPE_PROTO_FILE)
	mkdir -p $(GO_FILE_DIR)
	protoc -I . $< --go_out=.
