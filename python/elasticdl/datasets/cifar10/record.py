import numpy as np

N = 32

def encode(data, label):
    # 3 color channels, N * N image
    assert data.shape == (3, N, N) and data.dtype == "uint8"
    assert label >= 0 and label <= 9 and label.dtype == "uint8"
    return np.concatenate((data, label), axis=None).tobytes()

def decode(record):
    parsed = np.frombuffer(record, dtype="uint8")
    assert len(parsed) == 3 * N * N + 1
    label = parsed[-1]
    parsed = np.resize(parsed[:-1], new_shape=(3, N, N))
    return (parsed, label)

def show(data, label):
    """Print the image and label on terminal for debugging"""
    assert data.shape == (3, N, N) and data.dtype == "uint8"
    assert label >= 0 and label <= 9

    def pixel(r, g, b):
        return "\033[48;2;%d;%d;%dm" % (r, g, b) + " \033[0m"

    # Join r,g,b values
    data = np.stack(data, -1)

    for line in data:
        s = "".join(pixel(*x) for x in line)
        print(s)
    print("label =", label)
