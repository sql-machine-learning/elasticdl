import tensorflow as tf
from deepctr.layers.interaction import FM

from model_zoo.dac_ctr.utils import DNN, lookup_embedding_func


def deepfm_model(
    input_layers,
    dense_tensor,
    id_tensors,
    max_ids,
    sequence_tensor=None,
    deep_embedding_dim=8,
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
        sequence_tensor: 2-D tensor with string
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

    model = deepfm(
        input_layers,
        dense_tensor,
        linear_logits,
        deep_embeddings,
        deep_embedding_dim,
    )
    return model


def deepfm(
    input_layers,
    dense_tensor,
    linear_logits,
    deep_embeddings,
    deep_embedding_dim=8,
):
    # Deep Part
    dnn_input = tf.reshape(
        deep_embeddings, shape=(-1, len(deep_embeddings) * deep_embedding_dim)
    )
    if dense_tensor is not None:
        dnn_input = tf.keras.layers.Concatenate()([dense_tensor, dnn_input])
        linear_logits.append(
            tf.keras.layers.Dense(1, activation=None, use_bias=False)(
                dense_tensor
            )
        )

    # Linear Part
    if len(linear_logits) > 1:
        linear_logit = tf.keras.layers.Concatenate()(linear_logits)
    else:
        linear_logit = linear_logits[0]

    dnn_output = DNN(hidden_units=[16, 4], activation="relu")(dnn_input)
    dnn_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(
        dnn_output
    )

    if len(deep_embeddings) > 1:
        field_size = len(deep_embeddings)
        embeddings = tf.concat(
            deep_embeddings, 1
        )  # shape = (None, field_size, 8)
        embeddings = tf.reshape(embeddings, shape=(-1, field_size, 8))
        FM_output = FM(embeddings)
        # Output Part
        concat_input = tf.concat([linear_logit, dnn_logit, FM_output], 1)
    else:
        concat_input = tf.concat([linear_logit, dnn_logit], 1)

    logits = tf.reduce_sum(concat_input, 1, keepdims=True)
    probs = tf.reshape(tf.sigmoid(logits), shape=(-1,))
    return tf.keras.Model(
        inputs=input_layers,
        outputs={"logits": logits, "probs": probs},
        name="wide_deep",
    )
