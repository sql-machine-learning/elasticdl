cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(Swamp VERSION 0.0.1 DESCRIPTION "Swamp training system" HOMEPAGE_URL "" LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 11)

SET(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/coordinator/proto)
SET(PROTO_PY_DIR ${CMAKE_CURRENT_BINARY_DIR}/coordinator)
file(MAKE_DIRECTORY ${PROTO_PY_DIR})
file(WRITE ${PROTO_PY_DIR}/proto/__init__.py)

ADD_CUSTOM_COMMAND( OUTPUT ${PROTO_PY_DIR}/service_pb2_grpc.py ${PROTO_PY_DIR}/service_pb2.py
                   PRE_BUILD
				   COMMAND python -m grpc_tools.protoc -I ${CMAKE_CURRENT_SOURCE_DIR}/coordinator --python_out=${PROTO_PY_DIR} --grpc_python_out=${PROTO_PY_DIR} ${PROTO_DIR}/service.proto
				   DEPENDS ${PROTO_DIR}/service.proto
                   COMMENT " building coordinator python service" )

SET(ProtoGenPy ${PROTO_PY_DIR}/proto/__init__.py ${PROTO_PY_DIR}/service_pb2_grpc.py ${PROTO_PY_DIR}/service_pb2.py)
add_custom_target(coordinator DEPENDS ${ProtoGenPy})
