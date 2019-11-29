import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.ops import embedding_ops, math_ops


class SparseEmbedding(tf.keras.layers.Layer):
    """
    Input: indexes for the embedding entries with a shape of
      (batch_size, input_length). Input can be either dense tensor
      or SparseTensor.
    Output:
      corresponding (combined) embeddings with a shape of
      (batch_size, input_length, output_dim) if combiner is None
      (batch_size, output_dim) if combiner is not None
    Arguments:
      output_dim: the dimension of the embedding vector
      input_dim: the max input id. If 0 or None, will not check the range of
        input embedding ids.
      embeddings_initializer: Initializer for embedding table
      mask_zero: Whether or not the input value 0 is a special "padding"
        value that should be masked out.
        If input is SparseTensor, mask_zero must be False.
      input_length: Length of input sequences, when it is constant.
        This argument is required if you are going to connect
        `Flatten` then `Dense` layers upstream
        (without it, the shape of the dense outputs cannot be computed).
      combiner: A string specifying the reduction op or None if not used.
        "mean", "sqrtn" and "sum" are supported for the reduction op.
        If input is SparseTensor, combiner must set as a reduction op.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        embeddings_initializer="uniform",
        mask_zero=False,
        input_length=None,
        combiner=None,
        **kwargs
    ):
        if "input_shape" not in kwargs and input_length:
            kwargs["input_shape"] = (input_length,)
        super(SparseEmbedding, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero
        self.input_length = input_length
        self.combiner = combiner

        if self.combiner not in ["sum", "mean", "sqrtn"]:
            raise ValueError(
                "combiner must set sum, mean or sqrtn for sparse input"
            )

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name="embeddings",
            trainable=True,
        )
        self.built = True

    def call(self, inputs):
        # When saving a model with the layer, `tf.saved_model.save` will
        # feed the inputs with a Tensor not a SparseTensor, so we should
        # convert Tensor to `SparseTensor`.
        if not isinstance(inputs, tf.SparseTensor):
            idx = tf.where(tf.not_equal(inputs, 0))
            inputs = tf.SparseTensor(
                idx, tf.gather_nd(inputs, idx), (-1, self.input_dim)
            )

        dtype = K.dtype(inputs)
        if dtype != "int32" and dtype != "int64":
            inputs = math_ops.cast(inputs, "int32")
        out = embedding_ops.safe_embedding_lookup_sparse(
            embedding_weights=self.embeddings,
            sparse_ids=inputs,
            sparse_weights=None,
            combiner=self.combiner,
        )
        return out
