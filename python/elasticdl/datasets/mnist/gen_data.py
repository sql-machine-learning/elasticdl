from recordio.file import File
from tensorflow.python.keras.datasets import mnist, fashion_mnist
import record
import itertools
import argparse


def gen(file_base, data, label, *, chunk_size=4 * 1024, num_chunk=1024):
    assert len(data) == len(label) and len(data) > 0
    it = zip(data, label)
    try:
        for i in itertools.count():
            file_name = file_base + "-%04d" % i
            print("writing:", file_name)
            with File(file_name, "w", max_chunk_size=chunk_size) as f:
                for _ in range(num_chunk):
                    row = next(it)
                    f.write(record.encode(row[0], row[1]))
    except StopIteration:
        pass


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    gen("mnist-train", x_train, y_train)
    gen("mnist-test", x_test, y_test)

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    gen("fashion-train", x_train, y_train)
    gen("fashion-test", x_test, y_test)

