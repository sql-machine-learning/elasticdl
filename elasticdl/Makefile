PYTHON = python
PROTO_PATH=elasticdl/proto
GO_FILE_DIR = elasticdl/pkg/proto
CXX = g++
CAPI_DIR = elasticdl/pkg/kernel/capi

all: python_pb go_pb kernel

python_pb:
	$(PYTHON) -m grpc_tools.protoc -I${TF_PATH} -I. \
	    ${PROTO_PATH}/elasticdl.proto --python_out=. --grpc_python_out=.

go_pb:
	mkdir -p $(GO_FILE_DIR)
	mkdir -p ${GOPATH}/pkg/mod
	protoc -I${TF_PATH} ${TF_PATH}/tensorflow/core/framework/types.proto --go_out=${GOPATH}/pkg/mod
	protoc -I${TF_PATH} ${TF_PATH}/tensorflow/core/framework/resource_handle.proto --go_out=${GOPATH}/pkg/mod
	protoc -I${TF_PATH} ${TF_PATH}/tensorflow/core/framework/tensor_shape.proto --go_out=${GOPATH}/pkg/mod
	protoc -I${TF_PATH} ${TF_PATH}/tensorflow/core/framework/tensor.proto --go_out=${GOPATH}/pkg/mod
	protoc -I${TF_PATH} -I${PROTO_PATH} ${PROTO_PATH}/elasticdl.proto --go_out=plugins=grpc:.

kernel:
	${CXX} -O3 -std=c++11 ${CAPI_DIR}/kernel_api.cc -c -o ${CAPI_DIR}/libkernel_api.o
	ar crv ${CAPI_DIR}/libkernel_api.a ${CAPI_DIR}/libkernel_api.o
clean:
	rm -rf ${CAPI_DIR}/*.o ${CAPI_DIR}/*.a