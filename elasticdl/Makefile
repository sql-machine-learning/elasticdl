PYTHON = python

EDL_PROTO_FILE = elasticdl/proto/elasticdl.proto

GO_FILE_DIR = elasticdl/pkg/proto
GO_PB_FILE = $(GO_FILE_DIR)/elasticdl.pb.go

CXX = g++

all: python_pb $(GO_PB_FILE) kernel

python_pb:
	$(PYTHON) -m grpc_tools.protoc -I . $(EDL_PROTO_FILE) --python_out=. --grpc_python_out=.

$(GO_PB_FILE): $(EDL_PROTO_FILE)
	mkdir -p $(GO_FILE_DIR)
	protoc -I . $< --go_out=plugins=grpc:.

kernel:
	${CXX} -O3 -std=c++11 elasticdl/go/kernel/kernel.cc -I/usr/local/include/eigen3 \
	    -c -o elasticdl/go/kernel/kernel.o
	ar crv elasticdl/go/kernel/libkernel.a elasticdl/go/kernel/kernel.o
clean:
	rm -rf elasticdl/go/kernel/*.o elasticdl/go/kernel/*.a
