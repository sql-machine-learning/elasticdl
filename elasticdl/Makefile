PYTHON=python

elasticdl_pb:
	$(PYTHON) -m grpc_tools.protoc -I . elasticdl/proto/tensor_dtype.proto --python_out=.
	$(PYTHON) -m grpc_tools.protoc -I . elasticdl/proto/elasticdl.proto --python_out=. --grpc_python_out=.
