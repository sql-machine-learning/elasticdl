import tensorflow as tf


class TFExampleCodec(object):
    def encode(self, example):
        '''
        Take a tf example and return a string of encoded example
        '''
        return example.SerializeToString()

    def decode(self, raw, example_spec):
        '''
        Take an encoded string of tf example object and
        return a decoded tensor dict
        '''
        return tf.io.parse_single_example(raw, example_spec)
