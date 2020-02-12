PYTHON = python

TF_TENSOR_SHAPE_FILE = tensorflow/tensorflow/core/framework/tensor_shape.proto
EDL_PROTO_FILE = elasticdl/proto/elasticdl.proto

GO_FILE_DIR = elasticdl/pkg/proto
GO_PB_FILE = $(GO_FILE_DIR)/elasticdl.pb.go

CXX = g++
CAPI_DIR = elasticdl/pkg/kernel/capi

all: python_pb $(GO_PB_FILE) kernel

python_pb:
	$(PYTHON) -m grpc_tools.protoc -I . $(TF_TENSOR_SHAPE_FILE) --python_out=.
	$(PYTHON) -m grpc_tools.protoc -I . $(EDL_PROTO_FILE) --python_out=. --grpc_python_out=.

$(GO_PB_FILE): $(EDL_PROTO_FILE)
	mkdir -p $(GO_FILE_DIR)
	protoc -I . $< --go_out=plugins=grpc:.

kernel:
	${CXX} -O3 -std=c++11 ${CAPI_DIR}/kernel_api.cc -c -o ${CAPI_DIR}/libkernel_api.o
	ar crv ${CAPI_DIR}/libkernel_api.a ${CAPI_DIR}/libkernel_api.o
clean:
	rm -rf ${CAPI_DIR}/*.o ${CAPI_DIR}/*.a
