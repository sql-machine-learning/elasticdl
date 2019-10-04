import tensorflow as tf

from elasticdl.python.common.constants import Mode


def custom_model():
    inputs = tf.keras.layers.Input(shape=(4, 1), name="input")
    outputs = tf.keras.layers.Dense(3, name="output")(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="simple-model")


def loss(output, labels):
    return tf.reduce_sum(tf.reduce_mean(tf.reshape(output, [-1])) - labels)


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def dataset_fn(dataset, mode, metadata):
    def _parse_data(record):
        label_col_name = "class"

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

    if mode != Mode.PREDICTION:
        dataset = dataset.shuffle(buffer_size=200)
    return dataset


def eval_metrics_fn(predictions, labels):
    return {
        "dummy_metric": tf.reduce_sum(
            tf.reduce_mean(tf.reshape(predictions, [-1])) - labels
        )
    }
