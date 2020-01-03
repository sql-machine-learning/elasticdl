unittest_deps = [
    "@gtest//:gtest",
    "@gtest//:gtest_main",
    "@glog//:glog",
    "@com_github_gflags_gflags//:gflags",
]

cc_library(
    name = "tensor",
    srcs = ["tensor.cc"],
    hdrs = ["tensor.h"],
)


cc_test (
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = unittest_deps + [
        "//elasticdl/cpp/common:tensor",
    ],
)

