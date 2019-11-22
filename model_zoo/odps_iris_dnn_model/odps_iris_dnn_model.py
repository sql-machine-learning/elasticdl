import tensorflow as tf

from elasticdl.python.common.constants import Mode


def custom_model():
    inputs = tf.keras.layers.Input(shape=(4, 1), name="input")
    x = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(3, name="output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="simple-model")


def loss(labels, predictions):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf.cast(tf.reshape(labels, [-1]), tf.int32), predictions
        )
    )


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def dataset_fn(dataset, mode, metadata):
    def _parse_data(record):
        label_col_name = "class"
        record = tf.strings.to_number(record, tf.float32)

        def _get_features_without_labels(
            record, label_col_ind, features_shape
        ):
            features = [
                record[:label_col_ind],
                record[label_col_ind + 1 :],  # noqa: E203
            ]
            features = tf.concat(features, -1)
            return tf.reshape(features, features_shape)

        features_shape = (4, 1)
        labels_shape = (1,)
        if mode != Mode.PREDICTION:
            if label_col_name not in metadata.column_names:
                raise ValueError(
                    "Missing the label column '%s' in the retrieved "
                    "ODPS table." % label_col_name
                )
            label_col_ind = metadata.column_names.index(label_col_name)
            labels = tf.reshape(record[label_col_ind], labels_shape)
            return (
                _get_features_without_labels(
                    record, label_col_ind, features_shape
                ),
                labels,
            )
        else:
            if label_col_name in metadata.column_names:
                label_col_ind = metadata.column_names.index(label_col_name)
                return _get_features_without_labels(
                    record, label_col_ind, features_shape
                )
            else:
                return tf.reshape(record, features_shape)

    dataset = dataset.map(_parse_data)

    if mode == Mode.TRAINING:
        dataset = dataset.shuffle(buffer_size=200)
    return dataset


def eval_metrics_fn():
    return {
        "accuracy": lambda labels, predictions: tf.equal(
            tf.argmax(predictions, 1, output_type=tf.int32),
            tf.cast(tf.reshape(labels, [-1]), tf.int32),
        )
    }
