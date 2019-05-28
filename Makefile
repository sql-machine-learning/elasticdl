elasticdl_pb:
	python -m grpc_tools.protoc -I . elasticdl/python/elasticdl/proto/elasticdl.proto --python_out=. --grpc_python_out=.
