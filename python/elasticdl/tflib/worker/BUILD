package(default_visibility = ["//python:__subpackages__"])

py_library(
    name = "lib",
    srcs = [
        "worker.py",
    ],
)

py_test(
    name = "worker_test",
    srcs = [
        "worker_test.py",
    ],
    deps = [
        ":lib",
        "//python/elasticdl/tflib:lib",
        "//python/elasticdl/system:lib",
        "@pyrecordio//recordio:lib",
    ],
)
