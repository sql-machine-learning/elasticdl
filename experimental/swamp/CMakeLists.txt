cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(Swamp
        VERSION 0.0.1
                DESCRIPTION
                "Swamp training system"
                HOMEPAGE_URL
                ""
        LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 11)

set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/coordinator/proto)
set(PROTO_PY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/coordinator)

add_custom_command(
  OUTPUT ${PROTO_PY_DIR}/service_pb2_grpc.py ${PROTO_PY_DIR}/service_pb2.py
  PRE_BUILD
  COMMAND python -m grpc_tools.protoc -I ${CMAKE_CURRENT_SOURCE_DIR}/coordinator
          --python_out=${PROTO_PY_DIR}
          --grpc_python_out=${PROTO_PY_DIR} ${PROTO_DIR}/service.proto
  DEPENDS ${PROTO_DIR}/service.proto
  COMMENT " building coordinator python service")

set(ProtoGenPy ${PROTO_PY_DIR}/service_pb2_grpc.py
    ${PROTO_PY_DIR}/service_pb2.py)
add_custom_target(coordinator DEPENDS ${ProtoGenPy})
