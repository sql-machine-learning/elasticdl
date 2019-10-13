import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add
from elasticdl.python.common.constants import Mode


def custom_model():
    sparse_input = Input(shape=(100,), dtype='int64',
                         sparse=True, name='sparse_feature')
    dense_input = Input(shape=(4,), name='dense_feature')
    deep_embedding_column = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket(
            "sparse_feature", 100, dtype=tf.int64), 4)
    deep_embedded = tf.keras.layers.DenseFeatures(
        [deep_embedding_column])({'sparse_feature': sparse_input})

    wide_embedding_column = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket(
            "sparse_feature", 100, dtype=tf.int64), 1)
    wide_embedded = tf.keras.layers.DenseFeatures(
        [wide_embedding_column]
    )({'sparse_feature': sparse_input})

    deep_input = tf.keras.layers.concatenate([deep_embedded, dense_input], 1)
    dense1 = Dense(64, activation='relu')(deep_input)
    dense2 = Dense(8, activation='relu')(dense1)
    logits = Dense(1)(dense2)
    output = Add()([wide_embedded, logits])
    pred_score = Dense(1, activation='sigmoid', name='output_score')(output)

    return tf.keras.Model(inputs=[dense_input, sparse_input],
                          outputs=pred_score, name='wide-deep')


def loss(output, labels):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(tf.reshape(labels, (-1, 1)), tf.float32),
            logits=output
        )
    )


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def dataset_fn(dataset, mode, metadata):
    def _parse_data(record):
        dense_data = tf.strings.split([record[0]], sep=',')
        dense_data = tf.strings.to_number(dense_data, tf.float32)[0]
        sparse_data = tf.strings.split([record[1]], sep='\x01')
        sparse_data = tf.strings.to_number(sparse_data, tf.int64)[0]
        indices = tf.range(0, tf.size(sparse_data), dtype=tf.int64)
        indices = tf.reshape(indices, shape=(-1, 1))
        sparse_data = tf.sparse.SparseTensor(
            indices, sparse_data, dense_shape=(100,))
        label = tf.strings.to_number(record[2], tf.int32)
        features = {'sparse_feature': sparse_data, 'dense_feature': dense_data}
        return features, label

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
