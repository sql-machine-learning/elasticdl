PYTHON = python

TF_PATH = /Users/qijun/tensorflow
PROTO_PATH=elasticdl/proto

GO_FILE_DIR = elasticdl/pkg/proto

CXX = g++
CAPI_DIR = elasticdl/pkg/kernel/capi

all: python_pb go_pb kernel

python_pb:
	$(PYTHON) -m grpc_tools.protoc -I${TF_PATH} -I${PROTO_PATH} \
	    ${PROTO_PATH}/elasticdl.proto --python_out=${PROTO_PATH} --grpc_python_out=${PROTO_PATH}

go_pb: $(EDL_PROTO_FILE)
	mkdir -p $(GO_FILE_DIR)
	protoc -I${TF_PATH} ${TF_PATH}/tensorflow/core/framework/resource_handle.proto
	protoc -I${TF_PATH} ${TF_PATH}/tensorflow/core/framework/tensor_shape.proto --go_out=${GOPATH}/src
	protoc -I${TF_PATH} ${TF_PATH}/tensorflow/core/framework/tensor.proto --go_out=${GOPATH}/src
	protoc -I${TF_PATH} -I${PROTO_PATH} ${PROTO_PATH}/elasticdl.proto --go_out=plugins=grpc:.

kernel:
	${CXX} -O3 -std=c++11 ${CAPI_DIR}/kernel_api.cc -c -o ${CAPI_DIR}/libkernel_api.o
	ar crv ${CAPI_DIR}/libkernel_api.a ${CAPI_DIR}/libkernel_api.o
clean:
	rm -rf ${CAPI_DIR}/*.o ${CAPI_DIR}/*.a
