package(default_visibility = ["//python:__subpackages__"])

py_library(
    name = "lib",
    srcs = [
        "__init__.py",
        "ps.py",
        "ps_client.py",
    ],
)

py_test(
    name = "ps_test",
    srcs = [
        "ps_test.py"
    ],
    deps = [
        ":lib",
    ],
)

py_test(
    name = "ps_client_test",
    srcs = [
        "ps_client_test.py"
    ],
    deps = [
        ":lib",
    ],
)
