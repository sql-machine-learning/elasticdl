py_library(
    name = "record",
    srcs = [
        "record.py",
    ],
)

py_test(
    name = "record_test",
    srcs = [
        "record_test.py",
    ],
    deps = [
        ":record",
    ],
)

py_binary(
    name = "gen_data",
    srcs = [
        "gen_data.py",
    ],
    deps = [
        ":record",
    ],
)

py_binary(
    name = "show_data",
    srcs = [
        "show_data.py",
    ],
    deps = [
        ":record",
    ],
)
