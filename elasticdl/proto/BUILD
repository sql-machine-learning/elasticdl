load("@rules_proto//proto:defs.bzl", "proto_library")

cc_proto_library(
    name = "elasticdl_cc_proto",
    deps = [":elasticdl_proto"],
)

proto_library(
    name = "elasticdl_proto",
	srcs = ["elasticdl.proto", "tensor_dtype.proto"],
)