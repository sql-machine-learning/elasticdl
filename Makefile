master_pb:
	python -m grpc_tools.protoc -I . elasticdl/proto/master.proto --python_out=. --grpc_python_out=.
