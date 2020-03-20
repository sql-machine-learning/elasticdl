class TransformOp(object):
    def __init__(self, name, inputs, output):
        self.name = name

        # The array of input names
        self.inputs = inputs

        # The output name
        self.output = output


class Vocabularize(TransformOp):
    def __init__(self, name, inputs, output, vocabulary_list=None, vocabulary_file=None):
        super(Vocabularize, self).__init__(name, inputs, output)
        self.vocabulary_list = vocabulary_list
        self.vocabulary_file = vocabulary_file
        self.num_buckets = None


class Bucketize(TransformOp):
    def __init__(self, name, inputs, output, num_buckets=None, boundaries=None):
        super(Bucketize, self).__init__(name, inputs, output)
        self.num_buckets = num_buckets
        self.boundaries = boundaries


class Embedding(TransformOp):
    def __init__(self, name, inputs, output, dimension):
        super(Embedding, self).__init__(name, inputs, output)
        self.dimension = dimension


class Hash(TransformOp):
    def __init__(self, name, inputs, output, hash_bucket_size):
        super().__init__(name, inputs, output)
        self.hash_bucket_size = hash_bucket_size


class Concat(TransformOp):
    def __init__(self, name, inputs, output, id_offsets):
        super().__init__(name, inputs, output)
        self.id_offsets = id_offsets


class SchemaInfo(object):
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype
