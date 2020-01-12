unittest_deps = [
    "@gtest//:gtest",
    "@gtest//:gtest_main",
    "@glog//:glog",
    "@com_github_gflags_gflags//:gflags",
]

cc_library(
    name = "type",
    srcs = ["type.cc"],
    hdrs = ["type.h"],
    deps = ["@glog//:glog"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tensor",
    srcs = ["tensor.cc"],
    hdrs = ["tensor.h"],
    deps = ["//elasticdl/cc/common:type"],
    visibility = ["//visibility:public"],
)


cc_test (
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = unittest_deps + [
        "//elasticdl/cc/common:tensor",
    ],
)
