class Codec(object):
    def __init__(self):
        self._is_initialized = False

    def encode(self, data):
        raise("encode function is not implemented!")

    def decode(self, data):
        raise("decode function is not implemented!")
