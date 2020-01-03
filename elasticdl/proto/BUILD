load("@com_github_grpc_grpc//bazel:grpc_build_system.bzl", "grpc_proto_library")
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

proto_library(
    name = "elasticdl_proto",
	srcs = ["elasticdl.proto"],
	deps = ["@com_google_protobuf//:empty_proto"],
)

cc_proto_library(
    name = "elasticdl_cc_proto",
    deps = [":elasticdl_proto"],
)

cc_grpc_library(
    name = "elasticdl_cc_grpc",
    srcs = [":elasticdl_proto"],
    grpc_only = True,
    deps = [":elasticdl_cc_proto"],
)





