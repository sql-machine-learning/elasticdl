PYTHON = python

EDL_PROTO_FILE = elasticdl/proto/elasticdl.proto

GO_FILE_DIR = elasticdl/pkg/proto
CXX_FILE_DIR = elasticdl/pkg/c
GO_PB_FILE = $(GO_FILE_DIR)/elasticdl.pb.go

CXX = g++

all: python_pb $(GO_PB_FILE) kernel

python_pb:
	$(PYTHON) -m grpc_tools.protoc -I . $(EDL_PROTO_FILE) --python_out=. --grpc_python_out=.

$(GO_PB_FILE): $(EDL_PROTO_FILE)
	mkdir -p $(GO_FILE_DIR)
	protoc -I . $< --go_out=plugins=grpc:.

kernel:
	${CXX} -O3 -std=c++11 ${CXX_FILE_DIR}/kernel.cc -c -o ${CXX_FILE_DIR}/libkernel.o
	${CXX} -O3 -std=c++11 ${CXX_FILE_DIR}/vector.cc -c -o ${CXX_FILE_DIR}/libvector.o
	ar crv ${CXX_FILE_DIR}/libkernel.a ${CXX_FILE_DIR}/libkernel.o
	ar crv ${CXX_FILE_DIR}/libvector.a ${CXX_FILE_DIR}/libvector.o
clean:
	rm -rf ${CXX_FILE_DIR}/*.o ${CXX_FILE_DIR}/*.a
