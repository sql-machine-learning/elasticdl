cc_library(
    name = "tensor",
    srcs = ["tensor.cc"],
    hdrs = ["tensor.h"],
)

cc_test (
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "@gtest//:gtest",
        "@gtest//:gtest_main",
        "//elasticdl/cpp/common:tensor",
    ],
)

