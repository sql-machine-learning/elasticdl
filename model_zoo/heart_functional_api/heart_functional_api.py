import tensorflow as tf
from elasticdl.python.elasticdl.feature_column import feature_column

def get_feature_columns_and_inputs():
    feature_columns = []
    feature_input_layers = {}

    for header in ["trestbps", "chol", "thalach", "oldpeak", "slope", "ca"]:
        feature_columns.append(tf.feature_column.numeric_column(header))
        feature_input_layers[header] = tf.keras.Input(shape=(1,), name=header)

    age = tf.feature_column.numeric_column("age")
    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    )
    feature_columns.append(age_buckets)
    feature_input_layers["age"] = tf.keras.Input(shape=(1,), name="age")

    thal_hashed = tf.feature_column.categorical_column_with_hash_bucket(
        "thal", hash_bucket_size=10
    )
    thal_embedding = feature_column.embedding_column(
        thal_hashed, dimension=8
    )
    feature_columns.append(thal_embedding)
    feature_input_layers["thal"] = tf.keras.Input(
        shape=(1,), name="thal", dtype=tf.string
    )

    return feature_columns, feature_input_layers


def custom_model():
    feature_columns, feature_inputs = get_feature_columns_and_inputs()
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    x = feature_layer(feature_inputs)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    y = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(
        inputs=feature_inputs, outputs=y
    )

    return model


def loss(labels, predictions):
    labels = tf.reshape(labels, [-1])
    predictions = tf.reshape(predictions, [-1])
    return tf.keras.losses.binary_crossentropy(labels, predictions)


def optimizer(lr=1e-6):
    return tf.optimizers.SGD(lr)


def dataset_fn(dataset, mode, _):
    def _parse_data(record):

        feature_description = {
            "age": tf.io.FixedLenFeature([], tf.int64),
            "trestbps": tf.io.FixedLenFeature([], tf.int64),
            "chol": tf.io.FixedLenFeature([], tf.int64),
            "thalach": tf.io.FixedLenFeature([], tf.int64),
            "oldpeak": tf.io.FixedLenFeature([], tf.float32),
            "slope": tf.io.FixedLenFeature([], tf.int64),
            "ca": tf.io.FixedLenFeature([], tf.int64),
            "thal": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }

        parsed_record = tf.io.parse_single_example(record, feature_description)
        label = parsed_record.pop("target")

        return parsed_record, label

    dataset = dataset.map(_parse_data)

    return dataset


def eval_metrics_fn():
    return {
        "accuracy": lambda labels, predictions: tf.equal(
            tf.argmax(predictions, 1, output_type=tf.int32),
            tf.cast(tf.reshape(labels, [-1]), tf.int32),
        )
    }
