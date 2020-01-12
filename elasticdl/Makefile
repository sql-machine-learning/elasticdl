PYTHON = python

EDL_PROTO_FILE = elasticdl/proto/elasticdl.proto

GO_FILE_DIR = elasticdl/pkg/proto
GO_PB_FILE = $(GO_FILE_DIR)/elasticdl.pb.go

all: python_pb $(GO_PB_FILE)

python_pb:
	$(PYTHON) -m grpc_tools.protoc -I . $(EDL_PROTO_FILE) --python_out=. --grpc_python_out=.

$(GO_PB_FILE): $(EDL_PROTO_FILE)
	mkdir -p $(GO_FILE_DIR)
	protoc -I . $< --go_out=plugins=grpc:.
