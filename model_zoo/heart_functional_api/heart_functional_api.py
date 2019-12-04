import tensorflow as tf 

def get_feature_columns():
    feature_columns = []

    for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
        feature_columns.append(tf.feature_column.numeric_column(header))

    age = tf.feature_column.numeric_column('age')
    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    thal_hashed = tf.feature_column.categorical_column_with_hash_bucket('thal', hash_bucket_size=100)
    thal_embedding = tf.feature_column.embedding_column(thal_hashed, dimension=8)

    return feature_columns

def custom_model():
    feature_columns = get_feature_columns()
    dense_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    model = tf.keras.Sequential([
        dense_feature_layer,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model

def loss(labels, predictions):
    labels = tf.reshape(labels, [-1])
    return tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=predictions, labels=labels
        )
    )

def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)

def dataset_fn(dataset, mode, _):
    def _parse_data(record):
        return None

    dataset = dataset.map(_parse_data)

    return dataset