PYTHON = python

EDL_PROTO_FILE = elasticdl/proto/elasticdl.proto

GO_FILE_DIR = elasticdl/pkg/proto
GO_PB_FILE = $(GO_FILE_DIR)/elasticdl.pb.go

CXX = g++
CC_FILE_DIR = elasticdl/cc

all: python_pb $(GO_PB_FILE) optimizer_kernel

python_pb:
	$(PYTHON) -m grpc_tools.protoc -I . $(EDL_PROTO_FILE) --python_out=. --grpc_python_out=.

$(GO_PB_FILE): $(EDL_PROTO_FILE)
	mkdir -p $(GO_FILE_DIR)
	protoc -I . $< --go_out=plugins=grpc:.

optimizer_kernel:
	${CXX} -O3 -std=c++11 ${CC_FILE_DIR}/sgd.cc -I/usr/local/include/eigen3 -c -o sgd.o
	ar crv liboptimizer_kernel.a *.o
clean:
	rm -rf *.o *.a
