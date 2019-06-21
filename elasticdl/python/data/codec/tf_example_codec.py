import tensorflow as tf


class TFExampleCodec(object):
    '''
    Codec should be a stateless class. Any parameters needed by encode/decode
    should be passed in as the arguments of encode/decode function
    '''
    def encode(self, example):
        '''
        Take a tf example and return a string of encoded example
        '''
        return example.SerializeToString()

    def decode(self, raw):
        '''
        Take an encoded string of tf example object and
        return a decoded tensor dict
        '''
        example = tf.train.Example()
        example.ParseFromString(raw)
        return example
        # return tf.io.parse_single_example(raw, example_spec)


codec = TFExampleCodec()
