class Codec(object):
    def __init__(self):
        self._is_initialized = False

    def encode(self, data):
        """
        Take a {feature_name: ndarray} object and return the encoded data 
        TODO: This should be changed totake a {feature_name: example}
              once we implement https://github.com/wangkuiyi/elasticdl/pull/675
        """
        raise("encode function is not implemented!")

    def decode(self, data):
        """
        Take a sample of raw data and return a {feature_name: Tensor} 
        dictionary
        """
        raise("decode function is not implemented!")
