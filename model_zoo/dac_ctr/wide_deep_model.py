import tensorflow as tf


def wide_deep_model(
    input_layers,
    standardized_tensor,
    id_tensors,
    max_ids,
    deep_embedding_dim=8,
    dnn_hidden_units=[16, 4],
    dnn_activation="relu",
):
    """
    Args:
        input_layers: dict, the key is feature name and
            the value is tf.keras.Input.
        standardized_tensor:
        id_tensors: dict, the key is a string and the value
            is a tensor outputed by the transform function
        max_ids: dict, the key is group name and the value is the max
            integer id of this group.
        deep_embedding_dim: The output dimension of embedding layer for
            deep parts.
    """
    # wide part
    linear_logits = lookup_embedding_func(
        id_tensors, max_ids, embedding_dim=1,
    )

    # deep part
    deep_embeddings = lookup_embedding_func(
        id_tensors, max_ids, embedding_dim=deep_embedding_dim,
    )

    model = wide_deep(
        input_layers,
        standardized_tensor,
        linear_logits,
        deep_embeddings,
        deep_embedding_dim,
        dnn_hidden_units,
        dnn_activation,
    )
    return model


def wide_deep(
    input_layers,
    standardized_tensor,
    linear_logits,
    deep_embeddings,
    deep_embedding_dim=8,
    dnn_hidden_units=[16, 4],
    dnn_activation="relu",
):
    # Deep Part
    dnn_input = tf.reshape(
        deep_embeddings, shape=(-1, len(deep_embeddings) * deep_embedding_dim)
    )
    if standardized_tensor is not None:
        dnn_input = tf.keras.layers.Concatenate()(
            [standardized_tensor, dnn_input]
        )
        linear_logits.append(
            tf.keras.layers.Dense(1, activation=None, use_bias=False)(
                standardized_tensor
            )
        )

    # Linear Part
    if len(linear_logits) > 1:
        linear_logit = tf.keras.layers.Concatenate()(linear_logits)
    else:
        linear_logit = linear_logits[0]

    dnn_output = DNN(hidden_units=dnn_hidden_units, activation=dnn_activation)(
        dnn_input
    )
    dnn_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(
        dnn_output
    )

    # Output Part
    concat_input = tf.concat([linear_logit, dnn_logit], 1)

    logits = tf.reduce_sum(concat_input, 1, keepdims=True)
    probs = tf.reshape(tf.sigmoid(logits), shape=(-1,))
    return tf.keras.Model(
        inputs=input_layers,
        outputs={"logits": logits, "probs": probs},
        name="wide_deep",
    )


def lookup_embedding_func(
    id_tensors, max_ids, embedding_dim,
):
    """
    Args:
        id_layers: dict, the key is a string and
            the value is tf.keras.Input.
        standardized_tensor:
        input_tensors: dict, the key is a string and the value
            is a tensor outputed by the transform function
        max_ids: dict, the key is a string and the value is the max
            integer id of this group.
        deep_embedding_dim: The output dimension of embedding layer for
            deep parts.
    """

    embeddings = []
    for name, id_tensor in id_tensors.items():
        wide_embedding_layer = tf.keras.layers.Embedding(
            max_ids[name], embedding_dim
        )
        embedding = wide_embedding_layer(id_tensor)
        embedding_sum = tf.keras.backend.sum(embedding, axis=1)
        embeddings.append(embedding_sum)
    return embeddings


class DNN(tf.keras.layers.Layer):
    """ DNN layer

    Args:
        hidden_units: A list with integers
        activation: activation function name, like "relu"
    inputs: A Tensor
    """

    def __init__(self, hidden_units, activation=None, **kwargs):
        super(DNN, self).__init__(**kwargs)
        self.dense_layers = []
        if not hidden_units:
            raise ValueError("The hidden units cannot be empty")
        for hidden_unit in hidden_units:
            self.dense_layers.append(
                tf.keras.layers.Dense(hidden_unit, activation=activation)
            )

    def call(self, inputs):
        output = inputs
        for layer in self.dense_layers:
            output = layer(output)
        return output
