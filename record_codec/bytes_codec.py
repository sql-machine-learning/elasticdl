import numpy as np

N = 28


# TODO: maybe use TF variant tensor to do more flexible encoding.
class BytesCodec(object):
    def encode(self, data):
        values = [] 
        for _, f_value in data:
            values.append(f_value)
        return np.concatenate(values, axis=None).tobytes()

    def decode(self, record):
        parsed = np.frombuffer(record, dtype="uint8")
        assert len(parsed) == N * N + 1
        label = parsed[-1]
        parsed = np.resize(parsed[:-1], new_shape=(N, N))
        return {'image': parsed, 'label': label}

    def show(self, data, label):
        """Print the image and label on terminal for debugging"""
        assert data.shape == (N, N) and data.dtype == "uint8"
        assert label >= 0 and label <= 9

        def grey(x):
            return "\033[48;2;%d;%d;%dm" % (x, x, x) + " \033[0m"

        for line in data:
            s = "".join(grey(x) for x in line)
            print(s)
        print("label =", label)
