import tensorflow as tf
from deepctr.layers.interaction import CrossNet

from model_zoo.dac_ctr.utils import DNN, lookup_embedding_func


def dcn_model(
    input_layers, dense_tensor, id_tensors, max_ids, deep_embedding_dim=8,
):
    """
    Args:
        input_layers: dict, the key is feature name and
            the value is tf.keras.Input.
        dense_tensor: A 2-D tensor with float dtype
        id_tensors: dict, the key is a string and the value
            is a tensor with int64 dtype
        max_ids: dict, the key is group name and the value is the max
            integer id of this group.
        deep_embedding_dim: The output dimension of embedding layer for
            deep parts.
    """
    # linear part
    linear_logits = lookup_embedding_func(id_tensors, max_ids, 1)

    # deep part
    deep_embeddings = lookup_embedding_func(
        id_tensors, max_ids, deep_embedding_dim
    )

    model = dcn(
        input_layers,
        dense_tensor,
        linear_logits,
        deep_embeddings,
        deep_embedding_dim,
    )
    return model


def dcn(
    input_layers,
    dense_tensor,
    linear_logits,
    deep_embeddings,
    deep_embedding_dim=8,
    l2_reg=1e-5,
):
    # Deep Part
    dnn_input = tf.reshape(
        deep_embeddings, shape=(-1, len(deep_embeddings) * deep_embedding_dim)
    )
    if dense_tensor is not None:
        dnn_input = tf.keras.layers.Concatenate()([dense_tensor, dnn_input])
        linear_logits.append(
            tf.keras.layers.Dense(1, use_bias=False, activation=None)(
                dense_tensor
            )
        )

    if len(linear_logits) > 1:
        linear_logit = tf.keras.layers.Concatenate()(linear_logits)
    else:
        linear_logit = linear_logits[0]

    dnn_output = DNN(hidden_units=[16, 4], activation="relu")(dnn_input)

    cross_out = CrossNet(2, l2_reg=l2_reg)(dnn_input)
    # Output Part
    concat_input = tf.concat([dnn_output, cross_out], 1)
    deep_cross_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None
    )(concat_input)

    concat_input = tf.concat([linear_logit, deep_cross_logit], 1)
    logits = tf.reduce_sum(concat_input, 1, keepdims=True)

    probs = tf.reshape(tf.sigmoid(logits), shape=(-1,))
    return tf.keras.Model(
        inputs=input_layers,
        outputs={"logits": logits, "probs": probs},
        name="dcn",
    )
