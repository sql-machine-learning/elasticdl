package(default_visibility = ["//python:__subpackages__"])

py_library(
    name = "lib",
    srcs = [
        "master.py"
    ],
    deps = [
        "@pyrecordio//recordio:lib",
    ],
)

py_test(
    name = "master_test",
    srcs = [
        "master_test.py",
    ],
    deps = [
        ":lib",
    ],
)
